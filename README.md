# 📚 Goodreads, Bad Ratings 

## Overview  
What makes a book polarizing? This project explores the most disliked titles on Goodreads by analyzing over 15 million records of metadata and reader reviews. Using a two-phase approach—first with structured numerical data and then full-text review analysis—I examine the patterns behind one-star ratings, from genre bias to mismatched expectations and recurring complaint themes. The result is a rich combination of data storytelling, text mining, and visual interpretation that surfaces what readers love to hate—and why.

Negative reviews offer a rich, if often blunt, form of feedback. By focusing on one-star reviews—often emotionally charged and thematically rich—this project surfaces common patterns of disappointment and explores how reader expectations collide with genre, style, and content. This exploration serves as both a cultural critique and a foundation for more advanced modeling of user sentiment and experience.

---

## Workflow 

### Proof of Concept: Exploratory Analysis on Numerical Metadata

Before working with the full-text UCSD dataset, I conducted a preliminary analysis using a structured-only dataset from Hugging Face. This dataset contained book-level metadata (e.g., average star ratings, review counts, genres) but no full review text. This separate notebook and data source allowed me to develop and validate early hypotheses about patterns in dissatisfaction and prepare the logic for downstream tasks.

#### Key Goals

* Practice parsing, cleaning, and reshaping complex nested fields
* Visualize genre-level review patterns and star distributions
* Determine whether genre bias or engagement issues appeared in low-rated books

#### Key Cleaning and Feature Engineering Steps

```python
import ast  # Parse JSON-like strings to dict

# Convert 'community_reviews' from string to dictionary
sample_df['community_reviews'] = sample_df['community_reviews'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Extract 1-star review counts and percentages
sample_df['1_star_reviews_num'] = sample_df['community_reviews'].apply(
    lambda x: x['1_stars']['reviews_num'] if isinstance(x, dict) else 0)
sample_df['1_star_reviews_percentage'] = sample_df['community_reviews'].apply(
    lambda x: x['1_stars']['reviews_percentage'] if isinstance(x, dict) else 0)

# Parse genres
sample_df['genres'] = sample_df['genres'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Explode genres to one-per-row
sample_df = sample_df.explode('genres')

# Repeat above steps for 2–5 star review counts/percentages...
```

#### Exploratory Visualizations

1. **Top 20 Most Common Genres**
   Identified the most frequent genres in the dataset (excluding 'NaN').

   ```python
   genre_counts = sample_df['genres'].value_counts().head(20)
   sns.barplot(x=genre_counts.values, y=genre_counts.index)
   ```

2. **Genres with the Highest % of 1-Star Reviews**
   Averaged 1-star review percentages per genre to detect genre-based dissatisfaction.

3. **Number of Ratings vs. Star Rating**
   Scatterplot using log scale to investigate whether popularity correlates with positivity.

4. **Distribution of Star Ratings**
   Showed overall shape of rating spread (mostly skewed right).

5. **Correlation Matrix**
   Correlated key metrics (e.g., number of reviews, 1-star %s) to explore hidden relationships.

#### Outcome

This lightweight, metadata-driven exploration revealed early patterns: genres like romance and fantasy showed the most polarization, while average rating was not always a good predictor of satisfaction. These findings helped shape my final decision to focus on 1-star reviews and build a review-theme tagging system.

> ⚠️ Note: This work was done on a different dataset from Hugging Face and in a separate notebook. It did not feed into the full-text review pipeline but served as a critical methodological sandbox.

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


