var thumbs = document.querySelectorAll(".thumb");
var blobs = document.querySelectorAll(".blob");
var checkboxes = document.querySelectorAll("input[type='checkbox']");
var percentages = document.querySelectorAll(".percentage");
var equipLangBtn = document.getElementById("equip-lang");
var resetDistributionBtn = document.getElementById("reset-distribution");
var checkedIdx = [];


// Disable all button at start
equipLangBtn.disabled = true;
resetDistributionBtn.disabled = true;


// Get sessionStorage
if (sessionStorage.getItem("word-time") !== null) {
    document.getElementById("equip-lang").disabled = JSON.parse(sessionStorage.getItem("equip-lang"));
    document.getElementById("reset-distribution").disabled = JSON.parse(sessionStorage.getItem("reset-distribution"));

    document.getElementById("thumb1").value = sessionStorage.getItem("thumb1");
    document.getElementById("thumb2").value = sessionStorage.getItem("thumb2");
    document.getElementById("thumb3").value = sessionStorage.getItem("thumb3");
    document.getElementById("thumb4").value = sessionStorage.getItem("thumb4");
    document.getElementById("thumb5").value = sessionStorage.getItem("thumb5");

    checkedIdx = JSON.parse(sessionStorage.getItem("checkedIdx"));
    thumbs[checkedIdx[0]].style.visibility = "hidden";
    for (let i = 0; i < 5; i++) {
        if (checkedIdx.includes(i) === false) {
            thumbs[i].style.visibility = "hidden";
        }
    }

    document.getElementById("blob1").style.right = sessionStorage.getItem("blob1-right");
    document.getElementById("blob2").style.right = sessionStorage.getItem("blob2-right");
    document.getElementById("blob3").style.right = sessionStorage.getItem("blob3-right");
    document.getElementById("blob4").style.right = sessionStorage.getItem("blob4-right");
    document.getElementById("blob5").style.right = sessionStorage.getItem("blob5-right");

    document.getElementById("blob1").style.left = sessionStorage.getItem("blob1-left");
    document.getElementById("blob2").style.left = sessionStorage.getItem("blob2-left");
    document.getElementById("blob3").style.left = sessionStorage.getItem("blob3-left");
    document.getElementById("blob4").style.left = sessionStorage.getItem("blob4-left");
    document.getElementById("blob5").style.left = sessionStorage.getItem("blob5-left");

    document.getElementById("aus-cb").checked = JSON.parse(sessionStorage.getItem("aus-cb"));
    document.getElementById("ban-cb").checked = JSON.parse(sessionStorage.getItem("ban-cb"));
    document.getElementById("fin-cb").checked = JSON.parse(sessionStorage.getItem("fin-cb"));
    document.getElementById("rom-cb").checked = JSON.parse(sessionStorage.getItem("rom-cb"));
    document.getElementById("tur-cb").checked = JSON.parse(sessionStorage.getItem("tur-cb"));

    document.getElementById("aus-pr").value = sessionStorage.getItem("aus-pr");;
    document.getElementById("ban-pr").value = sessionStorage.getItem("ban-pr");;
    document.getElementById("fin-pr").value = sessionStorage.getItem("fin-pr");;
    document.getElementById("rom-pr").value = sessionStorage.getItem("rom-pr");;
    document.getElementById("tur-pr").value = sessionStorage.getItem("tur-pr");;
    percentages.forEach(percentage => {
        if (percentage.value === "0%") {
            percentage.style.visibility = "hidden";
        }
    });

    // word-length sementara ditiadakan dulu
    // document.getElementById("word-length").value = sessionStorage.getItem("word-length");
    document.getElementById("word-time").options[sessionStorage.getItem("word-time")].selected = true;
}


// When one of the lang button clicked, activate equipLangBtn
checkboxes.forEach(checkbox => {
    checkbox.addEventListener("click", function() {
        equipLangBtn.disabled = false;
    });
});
// When one of the slider changed, activate equipLangBtn
thumbs.forEach(thumb => {
    thumb.addEventListener("input", function() {
        resetDistributionBtn.disabled = false;
    });
});


// When you equip lang:
document.getElementById("equip-lang").addEventListener("click", function() {
    // Reset blob every time equip invoked
    blobs.forEach(blob => {
        blob.style.right = "100%";
        blob.style.left = "0%";
    });
    // Show/hide (or define/undefine value) thumb & percentages according to the checked/unchecked checkboxes.
    checkedIdx = [];
    for (let i = 0; i < 5; i++) {
        // For each CHECKED checkbox
        if (checkboxes[i].checked === true) {
            checkedIdx.push(i);
            // Hide first CHECKED thumb (because it won't do anything.)
            if (checkedIdx.length === 1) {
                thumbs[i].style.visibility = "hidden";
            } else {
                thumbs[i].style.visibility = "visible";
            }
            percentages[i].style.visibility = "visible";
        }
        // For each UNCHECKED checkbox
        else if (checkboxes[i].checked === false) {
            thumbs[i].style.visibility = "hidden";
            thumbs[i].value = 0;
            percentages[i].style.visibility = "hidden";
            percentages[i].value = "0%";
        }
    }
    // Thumbs and blob setup.
    for (let j = 0; j < checkedIdx.length; j++) {
        // Initialize thumb & percentages value after language query submitted.
        thumbs[checkedIdx[j]].value = (checkedIdx.length - j) / checkedIdx.length * 100;
        percentages[checkedIdx[j]].value = Math.round(100 / checkedIdx.length) + "%";
        // Set thumbs and blobs initial position after lang query submmited.
        if (j === 0) {
            blobs[checkedIdx[j]].style.right = "0%";
            if (checkedIdx.length === 1) {
                blobs[checkedIdx[j]].style.left = "0%";
            }
        } else if (j === checkedIdx.length - 1) {
            blobs[checkedIdx[j-1]].style.left = thumbs[checkedIdx[j]].value + "%";
            blobs[checkedIdx[j]].style.right = (100 - thumbs[checkedIdx[j]].value) + "%";
            blobs[checkedIdx[j]].style.left = "0%";
        } else {
            blobs[checkedIdx[j-1]].style.left = thumbs[checkedIdx[j]].value + "%";
            blobs[checkedIdx[j]].style.right = (100 - thumbs[checkedIdx[j]].value) + "%";
        }
    }
    // Save checkedIdx to sessionStorage so it can be used inter-functions.
    sessionStorage.setItem("checkedIdx", JSON.stringify(checkedIdx));
    // Disable equipLangBtn
    equipLangBtn.disabled = true;
    resetDistributionBtn.disabled = true;
});


// When you reset distribution:
document.getElementById("reset-distribution").addEventListener("click", function() {
    // checkedIdx = JSON.parse(sessionStorage.getItem("checkedIdx"));
    for (let j = 0; j < checkedIdx.length; j++) {
        // Initialize thumb & percentages value after language query submitted.
        thumbs[checkedIdx[j]].value = (checkedIdx.length - j) / checkedIdx.length * 100;
        percentages[checkedIdx[j]].value = Math.round(100 / checkedIdx.length) + "%";
        // Set thumbs and blobs initial position after lang query submmited.
        if (j === 0) {
            blobs[checkedIdx[j]].style.right = "0%";
            if (checkedIdx.length === 1) {
                blobs[checkedIdx[j]].style.left = "0%";
            }
        } else if (j === checkedIdx.length - 1) {
            blobs[checkedIdx[j-1]].style.left = thumbs[checkedIdx[j]].value + "%";
            blobs[checkedIdx[j]].style.right = (100 - thumbs[checkedIdx[j]].value) + "%";
            blobs[checkedIdx[j]].style.left = "0%";
        } else {
            blobs[checkedIdx[j-1]].style.left = thumbs[checkedIdx[j]].value + "%";
            blobs[checkedIdx[j]].style.right = (100 - thumbs[checkedIdx[j]].value) + "%";
        }
    }
});


// When you commit to make word, save visuals to sessionStorage
document.getElementById("make-word").addEventListener("click", function() {
    sessionStorage.setItem("equip-lang", equipLangBtn.disabled);
    sessionStorage.setItem("reset-distribution", resetDistributionBtn.disabled);

    sessionStorage.setItem("thumb1", document.getElementById("thumb1").value);
    sessionStorage.setItem("thumb2", document.getElementById("thumb2").value);
    sessionStorage.setItem("thumb3", document.getElementById("thumb3").value);
    sessionStorage.setItem("thumb4", document.getElementById("thumb4").value);
    sessionStorage.setItem("thumb5", document.getElementById("thumb5").value);

    sessionStorage.setItem("blob1-right", document.getElementById("blob1").style.right);
    sessionStorage.setItem("blob2-right", document.getElementById("blob2").style.right);
    sessionStorage.setItem("blob3-right", document.getElementById("blob3").style.right);
    sessionStorage.setItem("blob4-right", document.getElementById("blob4").style.right);
    sessionStorage.setItem("blob5-right", document.getElementById("blob5").style.right);

    sessionStorage.setItem("blob1-left", document.getElementById("blob1").style.left);
    sessionStorage.setItem("blob2-left", document.getElementById("blob2").style.left);
    sessionStorage.setItem("blob3-left", document.getElementById("blob3").style.left);
    sessionStorage.setItem("blob4-left", document.getElementById("blob4").style.left);
    sessionStorage.setItem("blob5-left", document.getElementById("blob5").style.left);

    sessionStorage.setItem("aus-cb", document.getElementById("aus-cb").checked);
    sessionStorage.setItem("ban-cb", document.getElementById("ban-cb").checked);
    sessionStorage.setItem("fin-cb", document.getElementById("fin-cb").checked);
    sessionStorage.setItem("rom-cb", document.getElementById("rom-cb").checked);
    sessionStorage.setItem("tur-cb", document.getElementById("tur-cb").checked);

    sessionStorage.setItem("aus-pr", document.getElementById("aus-pr").value);
    sessionStorage.setItem("ban-pr", document.getElementById("ban-pr").value);
    sessionStorage.setItem("fin-pr", document.getElementById("fin-pr").value);
    sessionStorage.setItem("rom-pr", document.getElementById("rom-pr").value);
    sessionStorage.setItem("tur-pr", document.getElementById("tur-pr").value);

    // sessionStorage.setItem("word-length", document.getElementById("word-length").value);
    sessionStorage.setItem("word-time", document.getElementById("word-time").options.selectedIndex);
});

// Set event to all slider, blobs, and percentages conditionally
thumbs.forEach(function(thumb, idx) {
    thumb.addEventListener("input", () => {
        // If thumb index is in checkedIdx, set thumb/blob behavior accordingly
        if (checkedIdx.includes(idx)) {
            let idxRelativeToChecked = checkedIdx.indexOf(idx);
            // Setup for first thumb in checkedIdx
            if (idxRelativeToChecked === 0) {
                thumb.value = Math.max(thumb.value, parseInt(thumbs[checkedIdx[1]].value) + 1);
            }
            // Setup for last thumb in checkedIdx
            else if (idxRelativeToChecked === checkedIdx.length - 1) {
                thumb.value = Math.max(1, Math.min(thumb.value, parseInt(thumbs[checkedIdx[idxRelativeToChecked - 1]].value) - 1));
                blobs[checkedIdx[idxRelativeToChecked - 1]].style.left = thumb.value + "%";
                blobs[checkedIdx[idxRelativeToChecked]].style.right = (100 - thumb.value) + "%";
                percentages[checkedIdx[idxRelativeToChecked - 1]].value = (parseInt(thumbs[checkedIdx[idxRelativeToChecked - 1]].value) - parseInt(thumb.value)) + "%";
                percentages[checkedIdx[idxRelativeToChecked]].value = thumb.value + "%";
            }
            // Setup for remaining checked thumb
            else {
                thumb.value = Math.min(Math.max(thumb.value, parseInt(thumbs[checkedIdx[idxRelativeToChecked + 1]].value) + 1), parseInt(thumbs[checkedIdx[idxRelativeToChecked - 1]].value) - 1);
                blobs[checkedIdx[idxRelativeToChecked - 1]].style.left = thumb.value + "%";
                blobs[checkedIdx[idxRelativeToChecked]].style.right = (100 - thumb.value) + "%";
                percentages[checkedIdx[idxRelativeToChecked - 1]].value =  (parseInt(thumbs[checkedIdx[idxRelativeToChecked - 1]].value) - parseInt(thumb.value)) + "%";
                percentages[checkedIdx[idxRelativeToChecked]].value =  (parseInt(thumb.value) - parseInt(thumbs[checkedIdx[idxRelativeToChecked + 1]].value)) + "%";
            }
        }
    });
});