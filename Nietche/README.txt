Step 1 Install requirements
pip install numpy scikit-learn spacy plotly textstat vaderSentiment
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_sm

Step 2 Run
python classifiers.py
python tfidf_only.py
python visuals_tfidf.py
python visuals_all_feat.py



-------------------------------------------
Withoud Data:
Step 1 Install requirements
pip install numpy scikit-learn spacy plotly textstat vaderSentiment
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_sm

Step 2 Prepare data with TextEdit.py (required)
Put your raw .txt books in a folder, for example rawtxt/.
Open TextEdit.py and set:
RAW_DIR → folder with your .txt books
CLEAN_DIR → output folder for cleaned paragraphs
TOKENIZED_DIR → output folder for tokenized paragraphs
Run: python TextEdit.py

Step 3 Build features
Ensure your raw books are in rawtxt/ and TextEdit.py already created CleanParagraphs/ and Tokenized_Paras/.
Run: python DataParsing.py

Step 4 Feature classifiers
Open classifiers.py and set:
textfeatures_dir → your TextFeatures/ folder
option: choose label_chaps_to_periods3.json or label_chaps_to_periods.json for three or four period classification
Run: python classifiers.py

Step 5 TF-IDF baselines
In tfidf_only.py, set TOKENIZED_DIR if your Tokenized_Paras/ is elsewhere.
Run: python tfidf_only.py

Step 6 Visualizations
In visuals_all_feat.py, set:
FEATURES → your TextFeatures/
LABELS → the labels JSON you chose in Step 4
Run: python visuals_all_feat.py
In visuals_tfidf.py, set TOKENIZED_DIR if needed.
Run: python visuals_tfidf.py
