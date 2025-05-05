# 📚 Goodreads, Bad Ratings 

## Overview  
What makes a book polarizing? This project explores the most disliked titles on Goodreads by analyzing over 15 million records of metadata and reader reviews. Using a two-phase approach—first with structured numerical data and then full-text review analysis—I examine the patterns behind one-star ratings, from genre bias to mismatched expectations and recurring complaint themes. The result is a rich combination of data storytelling, text mining, and visual interpretation that surfaces what readers love to hate—and why.

Negative reviews offer a rich, if often blunt, form of feedback. By focusing on one-star reviews—often emotionally charged and thematically rich—this project surfaces common patterns of disappointment and explores how reader expectations collide with genre, style, and content. This exploration serves as both a cultural critique and a foundation for more advanced modeling of user sentiment and experience.

---

## Workflow 

### Goodreads_concept: Exploratory Analysis on Numerical Metadata

Before working with the full-text UCSD dataset, I conducted a preliminary analysis using a dataset from Hugging Face. This dataset, available at [BrightData/Goodreads-Books](https://huggingface.co/datasets/BrightData/Goodreads-Books), contains book-level metadata such as:
	•	Average star ratings
	•	Community review breakdowns
	•	Genre tags
	•	Review and rating counts

Although this dataset lacks full-text reviews, it allowed me to prototype feature engineering steps, identify genre-level dissatisfaction trends, and validate early hypotheses about user engagement.

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

⸻

### Phase 1: Full Review + Metadata Extraction (UCSD Goodreads Dataset)

After validating core patterns through earlier metadata exploration (Goodreads_concept), I moved to the full 15+ million review dataset provided by UCSD. This dataset offered both structured book metadata and full review text, allowing me to scale up my analysis and begin natural language-based pattern detection.

1. Data Extraction and Transformation

The raw data was stored in .json.gz format and needed to be parsed line-by-line due to size. I used Python’s gzip and json libraries to iteratively load the data and write it to a flat CSV file in chunks:

```python
import gzip, json
import pandas as pd

chunk_size = 10000
chunks = []

with gzip.open("./Data/goodreads_reviews_dedup.json.gz", "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        chunks.append(json.loads(line))
        if (i + 1) % chunk_size == 0:
            df_chunk = pd.DataFrame(chunks)
            df_chunk.to_csv("goodreads_reviews.csv", mode="a", index=False, header=(i < chunk_size))
            chunks = []

    # Final batch
    if chunks:
        df_chunk = pd.DataFrame(chunks)
        df_chunk.to_csv("goodreads_reviews.csv", mode="a", index=False, header=False)
```
This process was repeated for goodreads_books.json.gz, and the two resulting DataFrames were merged on the book_id key.

2. Cleaning and Preprocessing

The merged dataset contained a wide range of features, many of which were irrelevant for my text-focused analysis. I dropped unnecessary fields to reduce noise and optimize performance, particularly for memory management.

```python
df_merged = df_reviews.merge(df_books, on="book_id", how="inner")
df_merged = df_merged.drop(columns=[
    'user_id', 'date_added', 'read_at', 'started_at', 'date_updated',
    'kindle_asin', 'work_id', 'n_comments', 'asin', 'similar_books',
    'publication_month', 'publication_day', 'edition_information', 'is_ebook'
])
df_merged = df_merged.dropna(subset=['review_text', 'description'])
```

Due to the large size of the full dataset (~30GB), I split the data into separate CSV files by star rating for more efficient downstream processing:

```python
for star in range(0, 6):
    df_star = df_merged[df_merged['rating'] == star]
    df_star.to_csv(f"./Data/{star}star_reviews.csv", index=False)
```
This modular approach made it easier to focus on specific subsets—in this case, I narrowed in on 1-star reviews, for now because the file size was small enough to work on from my laptop.

Perfect. Let’s outline and enhance Phase 2: Text-Based Theme Extraction and Review Classification, using your workflow and code from earlier.

⸻

Phase 2: Text-Based Theme Extraction and Review Classification (1-Star Sample)

This phase focuses on understanding why readers leave 1-star reviews by identifying recurring complaint themes in full-text review data. To ensure scalability, I used a 10,000-record sample (sample_1star) from the cleaned 1-star reviews CSV.

⸻

1. Initial Theme Assignment: Rule-Based Keyword Matching

To create interpretable categories for complaint analysis, I developed a dictionary of complaint themes, where each theme was associated with a list of indicative keywords.

```python
complaint_themes = {
    "Plot/Structure": ["slow", "boring", "drag", "confusing", "predictable"],
    "Character Issues": ["unlikeable", "flat", "annoying", "underdeveloped"],
    "Writing Style": ["cringe", "pretentious", "badly written"],
    "Expectations vs Reality": ["overhyped", "disappointed", "expected more"],
    ...
}
```
A custom assign_themes() function applied this dictionary to the cleaned review text, tagging each review with one or more relevant complaint themes.

```python
sample_1star['complaint_themes'] = sample_1star['review_clean'].apply(assign_themes)
```


⸻

2. Topic Discovery with NMF for Uncategorized Reviews

Reviews not matched to any theme were labeled “Uncategorized”. To improve theme coverage, I applied Non-negative Matrix Factorization (NMF) to those reviews. The process:
	•	Preprocessed reviews with lemmatization and stopword removal (combining NLTK, WordCloud, and custom lists).
	•	Vectorized text using TfidfVectorizer with n-grams (1–3).
	•	Trained a 7-topic NMF model to uncover common latent themes.
```python
nmf_model_uncat = NMF(n_components=7, random_state=42, max_iter=300)
nmf_model_uncat.fit(tfidf_matrix_uncat)
```
I interpreted the top words from each topic to expand and refine my original complaint_themes dictionary.

⸻

3. Updated Theme Assignment and Simplification

With a new and more comprehensive dictionary (complaint_themes_updated), I re-ran the theme assignment logic:
```python
sample_1star['complaint_themes_updated'] = sample_1star['review_clean'].apply(assign_themes_updated)
```
To enable simpler visualizations, I also created a top_theme column that extracts the first theme from each review’s list:
```python
sample_1star['top_theme'] = sample_1star['complaint_themes_updated'].apply(lambda x: x[0] if isinstance(x, list) and x else 'Uncategorized')
```
This allows each review to be visualized by a primary complaint category, even when multiple issues are present.

⸻

4. Visualizing Theme Frequency and Distribution

To surface macro trends in reader dissatisfaction, I created several visualizations:
	•	Bar chart of most common complaint themes
	•	Scatterplots showing how themes intersect with average Goodreads ratings and ratings count
	•	Theme breakdowns over time, genre, and publisher (planned)

Each chart used a custom Set2-style palette for visual accessibility and clarity.
```python
theme_palette = {
    'Engagement': '#66c2a5',
    'Plot/Structure': '#fc8d62',
    'Character Issues': '#8da0cb',
    ...
}
```

⸻

Exploratory Questions and Key Insight

With the refined complaint_themes_updated and top_theme columns, I began addressing high-level research questions around dissatisfaction patterns in Goodreads reviews:

Research Questions:
	•	What are the most common reasons people leave 1-star reviews?
	•	How do complaint themes differ across genres, publishers, and time periods?
	•	Do some books receive high average ratings despite negative feedback?
	•	How often do official book descriptions contradict user sentiment?
	•	When were the most 1-star reviewed books published?

⸻

Featured Insight: When Were the Most 1-Star Rated Books Published?

To explore how review trends evolved over time, I visualized the number of books receiving 1-star reviews by their publication year.
```python
# Filter for valid publication years
sample_1star_clean = sample_1star[(sample_1star['publication_year'] >= 1990) & 
                                  (sample_1star['publication_year'] <= 2018)]

# Count and fill missing years
year_range = list(range(1990, 2018))
year_counts = sample_1star_clean['publication_year'].value_counts().sort_index()
year_counts = year_counts.reindex(year_range, fill_value=0)


# Highlight spike years
highlight_years = [2011, 2012, 2013]
colors = ['orange' if year in highlight_years else 'gray' for year in year_counts.index]

# Plot
plt.figure(figsize=(14, 6))
bars = plt.bar(year_counts.index, year_counts.values, color=colors)

# Label spike years
for bar in bars:
    year = int(bar.get_x() + bar.get_width() / 2)
    height = bar.get_height()
    if year in highlight_years:
        plt.text(bar.get_x() + bar.get_width() / 2, height + 5,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10, weight='bold')

plt.title("When Were the Most 1-Star Rated Books Published?", fontsize=16, weight='bold')
plt.xlabel("Publication Year")
plt.ylabel("Number of 1-Star Reviews")
plt.xticks(year_range, rotation=45, fontsize=9)
plt.ylim(0, year_counts.max() + 80)
plt.figtext(0.5, -0.05,
            "Note: Spike between 2011–2013 may reflect changes in Goodreads review activity or publishing trends.",
            wrap=True, horizontalalignment='center', fontsize=10)
plt.tight_layout()
plt.grid(False)
plt.show()
```

📊 See: [when_were_1star_reviews_published.jpg](path)

Takeaway:
There was a notable spike in 1-star-reviewed books between 2011 and 2013. While the cause isn’t definitive, it may reflect Goodreads platform growth, shifting reader expectations, or changes in how books were marketed during that period.


File Structure
goodreads-bad-ratings/
│
├── Data/
│   ├── 1star_reviews.csv
│   ├── cleaned_numerical_data_only.csv  # Phase 0
│   ├── (Other .csvs for star ratings excluded via .gitignore)
│
├── notebooks/
│   ├── Goodreads_concept.ipynb          # Phase 0: Metadata-only EDA
│   ├── goodreads_sample1star.ipynb      # Sample review text analysis (10k)
│   └── Goodreads.ipynb                  # Full pipeline for 15M+ dataset
│
├── README.md
├── themes_1star_reviews.jpg             # Example visualization
└── other_figures/                       # Visual exports for portfolio


Next Steps & Limitations
This project currently focuses on 1-star reviews and theme detection in a 10k-sample dataset. Genre scraping and full dataset analysis are in progress. Due to dataset size, star rating chunking was required. Results may evolve with full-scale modeling.


