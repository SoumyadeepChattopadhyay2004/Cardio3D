# Cardio3D demo video: script and shot list

**Target length:** about 7.5 minutes (the brief asks for 3 to 10). I could not record or upload the video for
you, so this is a script to read and record yourself.
**Format:** screen recording with voice-over, 1080p, browser at full width.

**Before you record**
1. `uvicorn cardio.serve:app --port 8000`, then open <http://localhost:8000> and wait for the heart to appear.
2. Close other tabs and notifications. Zoom the browser to 100%.
3. Do one dry run. Say the numbers you see on screen, not the ones below: a retrained model gives slightly
   different values. The figures below are from the shipped artifacts.

Spoken lines are written to sound natural. Reword freely, but keep the facts.

---

## 0:00 to 0:40 | Why this exists

**Show:** the landing view, red-yellow disclaimer bar in frame.

> "Hi, this is Cardio3D. Risk calculators usually give you one number with no context. We wanted something you
> can look at and question. It predicts coronary artery disease and whether each of the three main arteries is
> narrowed, colours those arteries on a 3D heart, and shows which measurements drove each answer."

> "One thing first, visible the whole time: this is decision support and education only. It does not replace
> angiography or a doctor."

Point at the top banner, then scroll to show the footer.

## 0:40 to 2:00 | Data and validation, in plain terms

**Show:** the **Performance** tab.

> "It's trained on the UCI Extension of the Z-Alizadeh Sani dataset: 303 patients and 55 inputs covering
> history, examination, ECG, blood tests and echo."

> "Four separate models: overall CAD, then LAD, LCX and RCA. The LAD, LCX, RCA and Cath columns are never
> inputs, because they contain the answer. The code asserts that, and tests check it."

> "For validation we used nested, repeated cross-validation over all 303 patients. Every prediction you see in
> these tables comes from a model that never saw that patient, and model selection happens inside the folds, so
> the scores aren't flattered by picking the best model on the same data. Probabilities are calibrated, and
> intervals are bootstrap."

Read the AUC row: CAD about 0.90, LAD about 0.81, LCX about 0.72, RCA about 0.69. Point at the interval under each.

> "Honest reading: CAD and LAD are informative. LCX and RCA are weak, around 0.7. We tried several other model
> types and none lifted them, so that's the data's limit. Each target has its own decision threshold, listed
> here, because a plain 0.5 would miss most LCX and RCA cases."

Scroll to the ROC and calibration charts and the threshold table.

## 2:00 to 3:30 | Feature input workflow

**Show:** the **Prediction** tab. Open **Load an example patient**.

> "I can load a real patient from the dataset. The form is built from the data itself, grouped into
> demographics, risk factors, examination, ECG, labs and echo, with units and a tooltip explaining each measure."

Choose the first example. Wait for the heart to recolour. Read the line under the drop-down.

> "This line shows what catheterisation actually found, next to an out-of-fold estimate: what a model that never
> saw this patient would say. The live prediction is in-sample for dataset patients, so it can look better, and
> the app says so."

Open **History & risk factors**. Change **Typical Chest Pain**, then **Age**. Watch the percentages move.

> "Edits update the heart and the explanation a moment after I stop typing."

Type **150** into Age.

> "A value outside anything the model has seen produces a warning instead of false confidence. Blank fields are
> imputed, and it tells me how many."

Click **Reset**.

## 3:30 to 5:10 | The 3D heart

**Show:** the viewer, full attention.

> "This is real anatomy from the BodyParts3D database: ribcage, spine, the chambers and great vessels, and the
> three coronary arteries, each made of many named segments. Green means low predicted probability of narrowing,
> red means high."

Drag to rotate. Scroll to zoom. Click **Back**, **Left**, **Front**. Toggle **Skeleton** off and on.

Hover over an artery.

> "Hovering names the exact segment, here a marginal branch of the right coronary artery, and gives that
> vessel's probability."

Click the LAD.

> "Clicking an artery selects that vessel, and the explanation on the right switches to it. I can also use the
> cards underneath. Each card shows the probability and that target's cut-off."

Click the **ventricles**.

> "I can select anatomical regions too. The ventricle card says which arteries supply it, shows their
> predicted probabilities, and pulls in this patient's ejection fraction and wall-motion finding with their share
> of the explanation."

Click an atrium, then the aorta, then empty space (returns to overall CAD). Load a CAD-negative example and show
the colours change.

> "One honest caveat: the dataset doesn't say where along a vessel a narrowing is, so each artery is one colour
> for the whole vessel. It says how likely the vessel is narrowed, not where."

## 5:10 to 6:30 | Why did it say that?

**Show:** the right-hand panel.

> "For any target you get a sentence first: what pushes the risk up, what pulls it down, with the patient's
> actual values. Below it, the bars are SHAP contributions. Right raises risk, left lowers it, with each
> measurement's share."

Hover a row to show the tooltip.

> "Each measure has a plain-language description. Yes/no text columns are merged so each measurement appears once.
> Units are stated: log-odds for some models, probability for forests."

Click **LAD**, **LCX**, **RCA** in the selector. Click **Show all 55 measurements**, then collapse.

Open the **Importance** tab.

> "This tab shows what the model relies on in general, averaged over 150 patients. For overall CAD, typical chest
> pain leads, then age and the echo wall-motion finding."

Load the example where the model is confidently wrong (the third or fourth sample: a false alarm and a miss).

> "And a case where it's wrong. That's why this is decision support, not a diagnosis."

## 6:30 to 7:30 | How it's built

**Show:** the editor; keep each file up for a few seconds.

> "`data.py` and `train.py` load the data, remove the leakage columns, and run the nested cross-validation,
> calibration and bootstrap. Everything sits inside scikit-learn pipelines so nothing leaks between folds."

Show `LEAKAGE_KEYS` and `candidates()`.

> "`serve.py` is a small FastAPI service. It loads the bundle once and answers `/api/predict` in about a tenth
> of a second. `explain.py` holds the SHAP code shared by training and serving."

Show `serve.py` `predict`.

> "The front end is plain JavaScript and Three.js, vendored locally so it works offline. `anatomy.js` is the one
> place that says which mesh groups are which vessel or region, so adding anatomy doesn't mean touching the
> viewer. One function, `paintVessel`, turns a probability into a colour, so the cards and the 3D view can't
> disagree."

Show `anatomy.js`, then briefly the test folder: "Twelve Python tests plus a headless-browser test that checks the
rendered artery colours against the probabilities, clicks arteries and regions, and measures rendering with
CPU-only graphics."

## 7:30 to 8:00 | Wrap-up

**Show:** the app with the disclaimer in view.

> "To recap: calibrated models with honest, nested validation; real anatomy coloured by the predictions; and
> explanations you can question. The limits are real: a small single-centre dataset, no outside validation, and
> weak LCX and RCA predictions. Please treat it as a teaching and decision-support tool. Setup steps are in the
> README. Thanks for watching."

---

## Checklist before uploading to YouTube

- [ ] Length is between 3 and 10 minutes.
- [ ] The disclaimer is visible at the start and the end.
- [ ] Every number you say matches your screen.
- [ ] No personal information in browser tabs, terminal history or file paths.
- [ ] Title: "Cardio3D: explainable coronary artery risk on an interactive 3D heart".
- [ ] Description credits the dataset (UCI Extension of Z-Alizadeh Sani) and BodyParts3D (CC BY 4.0).
- [ ] Visibility is Public or Unlisted as the submission requires; test the link in a private window.
- [ ] Paste the link into the README and the submission form.
