# 📚 Goodreads, Bad Ratings 

## Overview  
This project investigates **book reviews on Goodreads** to identify patterns of user dissatisfaction. Using a combination of review and book metadata, the aim is to explore how reader sentiment diverges from official book descriptions and set the stage for a future **sentiment analysis** pipeline.

---

## Workflow 

### 1. Data Extraction  
- Loaded `.json.gz` files using Python’s `gzip` and `json` libraries.  
- Parsed nested JSON into flattened dataframes.
- Exported into manageable `.csv` files for exploration and cleaning.

**df_reviews** example:
import pandas as pd
import json
import gzip

chunk_size= 10000
chunks= []

with gzip.open ("./Data/goodreads_reviews_dedup.json.gz", "rt", encoding="utf-8") as f:
    for i, line in enumerate(f): #read line by line
        chunks.append(json.loads(line)) #convert json to stionf dict

    #every chuck line, process data to write csv
        if (i + 1) % chunk_size == 0:
            df_chunk = pd.DataFrame(chunks)
            df_chunk.to_csv("goodreads_reviews", mode="a", index= False, header = (i < chunk_size))
            chunks = []
        
if chunks:
    df_chunk = pd.DataFrame(chunks)
    df_chunk.to_csv("goodreads_reviews", mode ="a", index=False, header=False) 
    
### 2. Data Cleaning & Preprocessing  
#### Merging  
Merged `reviews` and `books` datasets on `book_id`:

df_merged = df_reviews.merge(df_books, on="book_id", how="inner")

Column Reduction

Dropped redundant or irrelevant columns to reduce noise and optimize memory usage:

df_merged = df_merged.drop(columns=[
    'user_id', 'date_added', 'read_at', 'started_at', 'date_updated',
    'kindle_asin', 'work_id', 'n_comments', 'asin', 'similar_books',
    'publication_month', 'publication_day', 'edition_information', 'is_ebook'
])

Duplicate & Null Checks
	•	Verified no duplicate review IDs
	•	Dropped rows missing review_text or description

df_merged = df_merged.dropna(subset=['review_text', 'description'])



3. Data Subsetting

To better manage the large dataset (~30GB), data was split by star rating (0–5):

for star in range(0, 6):
    df_star = df_merged[df_merged['rating'] == star]
    df_star.to_csv(f"./Data/{star}star_reviews.csv")

Example load and inspection:

df_5star = pd.read_csv("./Data/5star_reviews.csv")
df_5star.info()



⸻

Exploratory Focus
	•	Targeting 1-star reviews to identify trends in dissatisfaction
	•	Planning genre scraping via Goodreads URLs to enhance metadata
	•	Laying the groundwork for NLP-driven sentiment analysis and future predictive modeling

⸻

File Structure

goodreads-bad-ratings/
│
├── Data/
│   ├── 0star_reviews.csv
│   ├── 1star_reviews.csv
│   └── ... through 5star_reviews.csv
│
├── notebooks/
│   └── Goodreads.ipynb
│


Notes
	•	The dataset is very large, so chunking by rating is necessary for smoother computation.
	•	Genre data is incomplete – scraping in progress using Goodreads book URLs.
	•	This README will be updated with results from text analysis, sentiment modeling, and visualizations later in the semester.


