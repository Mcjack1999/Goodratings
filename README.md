# 📚 Goodreads, Bad Ratings 

## Overview  
What makes a book polarizing? This project explores the most disliked titles on Goodreads by analyzing over 15 million records of metadata and reader reviews. Using a two-phase approach—first with structured numerical data and then full-text review analysis—I examine the patterns behind one-star ratings, from genre bias to mismatched expectations and recurring complaint themes. The result is a rich combination of data storytelling, text mining, and visual interpretation that surfaces what readers love to hate—and why.

Negative reviews offer a rich, if often blunt, form of feedback. By focusing on one-star reviews—often emotionally charged and thematically rich—this project surfaces common patterns of disappointment and explores how reader expectations collide with genre, style, and content. This exploration serves as both a cultural critique and a foundation for more advanced modeling of user sentiment and experience.

---

## Workflow 

### 1. Data Extraction  
- Loaded `.json.gz` files using Python’s `gzip` and `json` libraries.  Link to [data source]([url](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html#datasets))
- Parsed nested JSON into flattened dataframes.
- Exported into manageable `.csv` files for exploration and cleaning.

**df_reviews** 

	import pandas as pd
	import json
	import gzip

	chunk_size= 10000
	chunks= []

	with gzip.open ("./Data/goodreads_reviews_dedup.json.gz", "rt", encoding="utf-8") as f:
   		 for i, line in enumerate(f): #read line by line
      		  chunks.append(json.loads(line)) #convert json to dict

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


