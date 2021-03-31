from flask import Flask, render_template, request, session
import pickle
import pandas as pd
import lang_maker


app = Flask(__name__)
app.secret_key = 'hello'

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


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        lang_input = {'aus': int(request.form['aus-pr'][:-1]) / 100,
                        'ban': int(request.form['ban-pr'][:-1]) / 100,
                        'fin': int(request.form['fin-pr'][:-1]) / 100,
                        'rom': int(request.form['rom-pr'][:-1]) / 100,
                        'tur': int(request.form['tur-pr'][:-1]) / 100}
        influence = list(lang_input.values())
        search_time = request.form['word-time']

        # Make Sentence
        sentence = lang_maker.generate_sentence(lang_digs, leng_dists, influence, search_time)
        session['sentence'] = sentence
        return render_template('index.html', sentence=sentence)
    else:
        if len(session) > 0:
            sentence = session['sentence']
            return render_template('index.html', sentence=sentence)
        else:
            sentence = "Result here."
            return render_template('index.html', sentence=sentence)


if __name__ == '__main__':
    app.run(debug=True)