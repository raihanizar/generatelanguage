import numpy as np
import pandas as pd
import re, pickle
from collections import Counter


score_tabl = pd.read_csv('/static/words/score_tabl_simplified.csv', index_col='Unnamed: 0', keep_default_na=False) # prevent string 'nan' interpreted as NaN
score_tabl2_3 = pd.read_csv('/static/words/score_tabl2_3.csv', index_col='Unnamed: 0', keep_default_na=False) # prevent string 'nan' interpreted as NaN
lang_digs = pd.read_csv('/static/words/lang_digs.csv', index_col='Unnamed: 0')
leng_dists = pd.read_csv('/static/words/leng_dists.csv', index_col='Unnamed: 0')
with open('/static/words/score_tabl_dict_simplified.pkl', 'rb') as f:
    score_tabl_dict = pickle.load(f)
with open('/static/words/score_tabl_dict2_3.pkl', 'rb') as f:
    score_tabl_dict2_3 = pickle.load(f)
with open('/static/words/char_dist_init.pkl', 'rb') as f:
    char_dist_init = pickle.load(f)
with open('/static/words/letter_distributions.pkl', 'rb') as f:
    letter_distributions = pickle.load(f)


def generate_sentence(lang_digs, leng_dists, infl, st):
    langs = ['aus', 'ban', 'fin', 'rom', 'tur']
    influence = infl
    if sum(influence) != 1: # kalo jumlah tidak 1 jumlah semuanya dengan residualnya bagi 5
        residual = 1 - sum(influence)
        influence = list(map(lambda x: x + residual/5, influence))
    search_time = st

    ''' Part 0 '''
    # Beginning
    langs += ['rand']
    influence += [0.05] # 'rand' set to 0.05 to encourage variance
    inputs = {l: i for (l, i) in zip(langs, influence)}

    # Digraph & length search preparation
    combine_digs = pd.concat([inputs[l] * lang_digs[l] for l in langs[:-1]], axis=1).sum(axis=1)
    combine_leng = pd.concat([inputs[l] * leng_dists[l] for l in langs[:-1]], axis=1).sum(axis=1)

    # Select 61 randomly from 201 best digs (for variability)
    selected_idx = np.random.choice(201, 81, replace=False)
    combine_digs_filtered = combine_digs.sort_values(ascending=False)[selected_idx]
    combine_digs_filtered = combine_digs_filtered + 500 # add 500 to smooth out digs probability
    combine_digs_norm = combine_digs_filtered / np.sum(combine_digs_filtered) # make probability sum to 1

    # Curate digs for len == 2 words
    combine_digs2 = combine_digs[[c for c in combine_digs.index if re.match(r"[aeiou]", c)]] # only if they had vocals
    combine_digs_filtered2 = combine_digs2.sort_values(ascending=False)[:150] # get best 150 (so each representation not too big)
    combine_digs_filtered2 = combine_digs_filtered2 + 5000 # add 5000 to smooth out digs probability
    combine_digs_norm2 = combine_digs_filtered2 / np.sum(combine_digs_filtered2) # make probability sum to 1

    # Ready to search for fresh digs!
    search_digs = combine_digs_norm.index
    search_prob = combine_digs_norm.values
    search_digs2 = combine_digs_norm2.index # for len == 2
    search_prob2 = combine_digs_norm2.values # for len == 2

    # Search!
    search_time_dict = {'quick': 12500, 'normal': 25000, 'long': 50000, 'longer': 100000}
    iteration = search_time_dict[search_time]


    ''' Part 1 '''
    # Generate words
    words = []
    word_lengths = np.random.choice([15] + list(combine_leng.index[1:]), iteration, p=combine_leng) # sementara len == 1 diganti len == 15
    for wl in word_lengths:
        if wl > 2: # kalau panjang kata > 2, normal
            search_length = wl // 2 + 1
            if wl % 2 == 0:
                truncate = 2
            else:
                truncate = 1
            word = ''.join(np.random.choice(search_digs, search_length, p=search_prob))[:-truncate]
            words.append(word)
        else: # kalau panjang kata < 2, pilih digraph yg ada vowelnya, untuk menghindari kata 2 huruf yg all-consonant
            word2 = np.random.choice(search_digs2, 1, p=search_prob2)[0]
            words.append(word2)

    # Buat tabel score
    word_scores = pd.DataFrame([score_word(word) if len(word) > 3 else score_word2_3(word) for word in words], columns=langs) # score buat yg len > 3
    word_distance_to_feel = word_scores.apply(compute_word_distance, args=(influence,), axis=1)

    # Buat tabel word-length-score
    df_words = pd.concat([pd.Series(words), pd.Series(word_lengths), word_distance_to_feel], axis=1)
    df_words.columns = ['word', 'length', 'score']


    ''' Part 2 '''
    # Ambil randomly 2000 * probability_occurence dari tiap length, sehingga menghasilkan total ~2000 word.
    filter_query = df_words.groupby(by='length') \
                            .apply(lambda x: x.nsmallest(int(2000 * combine_leng.loc[x.name]), columns='score'))
    filter_idx = filter_query.index.get_level_values(1)
    df_words_filtered = df_words.loc[filter_idx]

    # Itung plausibility (yg persebaran letter-nya masih dalam sigma 1.7 di standard distribution)
    filter_query2 = [abs(plausibility(w, langs, influence)) < 1.7 if len(w) > 3 else True for w in df_words_filtered['word']]
    df_words_filtered2 = df_words_filtered.loc[filter_query2]

    ## Get compactness score
    df_words_filtered2.loc[:, 'compactness'] = [compactness(score_word_table(w), langs, influence) if len(w) > 3 \
                                                    else compactness(score_word_table2_3(w), langs, influence) if len(w) > 1 \
                                                        else 0 for w in df_words_filtered2['word']]

    ## Ambil total ~100 best words. Dari tiap kelompok len ambil sesuai probability-nya
    df_words_filtered2_uniq = df_words_filtered2.drop_duplicates(subset=['word'])
    word_query3 = df_words_filtered2_uniq.groupby(by='length') \
                                            .apply(lambda x: x.nsmallest(int(100 * combine_leng.loc[x.name]), columns='compactness'))['word'].to_list()

    # Get index dari 100 best words, baru diapply ke df_words_filtered2 (biar distribusi kemunculan tetep terjaga)
    filter_query3 = []
    for w in word_query3:
        filter_query3 += df_words_filtered2[df_words_filtered2['word'] == w].index.to_list()
    df_words_filtered3 = df_words_filtered2.loc[filter_query3]


    ''' Part 3 '''
    # Probability sebuah word dimunculkan = probablity panjang word tsb / banyaknya kemunculan panjang word tsb
    chance_factor = Counter(df_words_filtered3['length'])
    df_words_filtered3.loc[:, 'chance'] = [combine_leng.loc[i] / chance_factor[i] for i in df_words_filtered3['length']]
    df_words_filtered3.loc[:, 'chance'] = df_words_filtered3['chance'] / np.sum(df_words_filtered3['chance'])

    # Membuat list word 50 kata
    word_in_sentence = 50
    word_sequence = np.random.choice(df_words_filtered3['word'], word_in_sentence, p=df_words_filtered3['chance'])


    ''' Part 4 '''
    # Persiapan generate sentence
    w2 = [w for w in word_sequence if len(w) == 2]
    set_w2 = set(w2)
    counter_w2 = Counter(w2)

    # Define definites-particles-conjuncts
    definites = [w for (w, v) in counter_w2.items() if v == max(counter_w2.values())] # yg muncul terbanyak (bisa lebih dari satu)
    set_w2 = set_w2 - set(definites) # apus tiap assignment
    particles = [w for w in set_w2 if re.match(r"^[aeiou]{1,}$", w)] # yg vokal semua / yg satu huruf (kalo muncul terbanyak tetep dianggap definites)
    set_w2 = set_w2 - set(particles) # apus tiap assignment
    conjuncts = list(set_w2) # sisanya

    # Pasang Definites, Conjuncts, dan Particles
    word_sequence_filtered = pd.Series([w for w in word_sequence if len(w) > 2])
    idx_used = []
    ## Definites
    for d in definites:
        idx = np.random.choice(list(set(word_sequence_filtered.index) - set(idx_used)), counter_w2[d], replace=False) # tag tempat yg belum diisi idx_used
        word_sequence_filtered[idx] = word_sequence_filtered[idx].apply(lambda x: d + ' ' + x)
    ## Conjuncts
    for c in conjuncts:
        idx = np.random.choice(list(set(word_sequence_filtered.index) - set(idx_used)), counter_w2[c], replace=False) # tag tempat yg belum diisi idx_used
        for i in idx:
            if np.random.random() < 0.3: # biar conjuncts ga kebanyakan, soalnya rawan banyak
                word_sequence_filtered[i] = word_sequence_filtered[i] + ' ' + c + ','
                idx_used += [i]
    ## Particles
    for p in particles:
        idx = np.random.choice(list(set(word_sequence_filtered.index) - set(idx_used)), counter_w2[p], replace=False) # tag tempat yg belum diisi idx_used
        for i in idx:
            r = np.random.random()
            if r < 0.3:
                word_sequence_filtered[i] = word_sequence_filtered[i] + ' ' + p + '?'
            else:
                word_sequence_filtered[i] = word_sequence_filtered[i] + ' ' + p[0] + '.'
        idx_used += idx.tolist()

    # Mitigasi kalimat yg terlalu sparse (gap antar tanda baca terlalu jauh)
    idx_used = [0] + sorted(idx_used) + [word_sequence_filtered.index[-1]]
    gaps = [idx_used[i + 1] - idx_used[i] for i in range(len(idx_used) - 1)]
    for i in range(len(idx_used) - 1):
        i1 = idx_used[i]
        i2 = idx_used[i + 1]
        gap = i2 - i1
        if gap > 7: # kalau gap-nya kejauhan (kasih titik tengah tengah)
            divisor = 3
            while gap / divisor >= 7:
                divisor += 1
            for d in range(1, divisor + 3): # lanjut hingga berhenti
                correct_pos = i1 + int(d * np.random.uniform(0.8, 1.1) * (gap // divisor))
                if correct_pos < i2: # jika belum lewat batas
                    r = np.random.random()
                    if r < 0.7:
                        word_sequence_filtered[correct_pos] = word_sequence_filtered[correct_pos] + '.' # Tambah titik
                    else:
                        word_sequence_filtered[correct_pos] = word_sequence_filtered[correct_pos] + ',' # Tambah koma

    # Post-process
    sentence = ' '.join(word_sequence_filtered)
    if np.argmax(influence) == 0: # kalau influence cenderung 'aus'
        sentence = re.sub(r"([a-zŋ]{3,})\s\1", r"\1-\1", sentence) # morphem > 3 huruf yang berulang ditambahin strip
        sentence = re.sub(r"([aeiou])\1", r"\1'\1", sentence) # vokal dobel dipisah apostrof
    else: # kalau yg lain
        sentence = re.sub(r"(\b[a-zŋ]+\b)\s\1", r"\1", sentence) # kata yang berulang dihapus pengulangannya
    sentence = re.sub(r"ŋ", r"ng", sentence) # Ganti 'ŋ' dengan 'ng'
    sentence = '. '.join([i[0].upper() + i[1:] for i in sentence.split('. ')]) # capitalize
    sentence = '? '.join([i[0].upper() + i[1:] for i in sentence.split('? ')]) # capitalize
    sentence = sentence + '.'
    sentence = re.sub(r"[\,\.]{2,}", ".", sentence) # correct full-stop
    return sentence


# Fungsi score word
def score_word(w):
    lw = len(w) - 2
    scores = sum([score_tabl_dict['_' + w[i:i+3]] if (i == 0) else score_tabl_dict[w[i:i+3] + '_'] if (i == lw-1) else score_tabl_dict[w[i:i+3]] for i in range(lw)])
    return scores / lw
def score_word2_3(w):
    return (score_tabl_dict2_3['_' + w[:2]] + score_tabl_dict2_3[w[-2:] + '_']) / 2


# Fungsi buat tabel skor dari word
def score_word_table(w):
    lw = len(w) - 2
    return score_tabl.loc[['_' + w[i:i+3] if (i == 0) else w[i:i+3] + '_' if (i == lw-1) else w[i:i+3] for i in range(lw)]]
def score_word_table2_3(w):
    return score_tabl2_3.loc[['_' + w[:2], w[-2:] + '_']]


# Fungsi buat hitung distance word fari target score
def compute_word_distance(row, infl):
    return sum([(row[l] - infl[l]) ** 2 for l in range(6)]) # mempertimbangkan 'rand'


# Compactness dari word
def compactness(word_tabl, langs, infl):
    return max([max(abs(word_tabl[key] - val)) for (key, val) in zip(langs, infl)])


# Plausibility
def plausibility(word, langs, infl):
    lw = len(word)
    letter_distribution = np.std(list({**char_dist_init, **Counter(word)}.values()))
    mean, std = np.sum([[v * infl for v in letter_distributions[lang][lw]] for (lang, infl) in zip(langs, infl)], axis=0)
    return (letter_distribution - mean) / std