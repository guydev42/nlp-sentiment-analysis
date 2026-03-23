"""
Generate a realistic synthetic product reviews dataset with 5,000 reviews
across 3 sentiment classes: positive, neutral, negative.
Reviews vary in length, vocabulary, and product category.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

N = 5000

# --- Product categories ---
categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports & Outdoors"]
category_probs = [0.28, 0.22, 0.20, 0.16, 0.14]
product_category = np.random.choice(categories, N, p=category_probs)

# --- Sentiment distribution: 45% positive, 30% neutral, 25% negative ---
sentiment = np.random.choice(
    ["positive", "neutral", "negative"], N, p=[0.45, 0.30, 0.25]
)

# --- Rating correlated with sentiment ---
rating = np.zeros(N, dtype=int)
for i in range(N):
    if sentiment[i] == "positive":
        rating[i] = np.random.choice([4, 5], p=[0.35, 0.65])
    elif sentiment[i] == "neutral":
        rating[i] = np.random.choice([2, 3, 4], p=[0.15, 0.60, 0.25])
    else:
        rating[i] = np.random.choice([1, 2], p=[0.55, 0.45])

# --- Review text generation ---

positive_openings = [
    "I absolutely love this product.",
    "Great purchase, very happy with it.",
    "This exceeded all my expectations.",
    "Fantastic quality for the price.",
    "Best purchase I have made in a long time.",
    "Really impressed with the build quality.",
    "Could not be happier with this item.",
    "This is exactly what I was looking for.",
    "Amazing product, works perfectly.",
    "Outstanding value, highly satisfied.",
    "Super pleased with my purchase.",
    "This product is a game changer for me.",
    "Very well made and sturdy.",
    "I bought this as a gift and they loved it.",
    "Excellent product all around.",
]

positive_details = [
    "The quality is excellent and it feels very durable.",
    "It arrived quickly and was packaged well.",
    "Easy to set up and use right out of the box.",
    "The color and design match the photos exactly.",
    "Works even better than advertised.",
    "My family uses it every day and it holds up well.",
    "The materials feel premium and well crafted.",
    "It fits perfectly and looks great.",
    "Customer service was helpful when I had a question.",
    "I have recommended this to several friends already.",
    "The instructions were clear and assembly took minutes.",
    "Performance has been consistent over several months.",
    "It does exactly what it says on the description.",
    "Good weight and balance, feels solid in hand.",
    "Battery life is much longer than I expected.",
    "The sound quality is crisp and clear.",
    "Very comfortable even after extended use.",
    "The price point is fair for what you get.",
]

positive_closings = [
    "Would definitely buy again.",
    "Highly recommend to anyone looking for this type of product.",
    "Five stars, no hesitation.",
    "I plan to buy another one as a gift.",
    "Overall a great buy.",
    "Very satisfied customer here.",
    "Will be ordering more soon.",
    "Cannot say enough good things about it.",
    "Totally worth the investment.",
    "A must-have in my opinion.",
]

neutral_openings = [
    "The product is okay for the price.",
    "It does what it is supposed to do.",
    "Decent product, nothing special.",
    "Average quality, meets basic needs.",
    "It works fine for everyday use.",
    "Not bad but not outstanding either.",
    "Functional product with some minor issues.",
    "It gets the job done.",
    "Reasonable product for casual use.",
    "Mixed feelings about this purchase.",
    "Middle of the road product.",
    "It is acceptable but I expected more.",
]

neutral_details = [
    "Some features are good but others feel lacking.",
    "The build quality is average compared to similar products.",
    "Setup took longer than expected but it works now.",
    "The size is slightly different from what was described.",
    "Performance is adequate for light use.",
    "It has a few quirks but nothing deal-breaking.",
    "The design is plain but functional.",
    "Quality seems fine for occasional use.",
    "Some parts feel a bit cheap but it holds together.",
    "It meets the basic requirements I had.",
    "The material could be better at this price point.",
    "Works as described though the finish is rough in spots.",
    "A few minor cosmetic flaws but nothing serious.",
    "The color is slightly different from the listing photos.",
]

neutral_closings = [
    "It serves its purpose.",
    "Would consider alternatives next time.",
    "Not sure if I would buy it again.",
    "Adequate for what I needed.",
    "Three stars feels right for this one.",
    "Could be improved but does the basics.",
    "Fair product, fair price.",
    "Nothing to complain about but nothing to praise either.",
]

negative_openings = [
    "Very disappointed with this product.",
    "Would not recommend this to anyone.",
    "This was a waste of money.",
    "Terrible quality, broke after a week.",
    "Not worth the price at all.",
    "I regret purchasing this item.",
    "Extremely frustrated with this product.",
    "Save your money and look elsewhere.",
    "This product fell apart almost immediately.",
    "Awful experience from start to finish.",
    "The product arrived damaged.",
    "Nothing like what was described.",
    "Cheaply made and overpriced.",
    "I expected much better quality.",
]

negative_details = [
    "The product broke after just a few uses.",
    "The material feels very cheap and flimsy.",
    "It does not fit as described and I cannot return it.",
    "Assembly instructions were confusing and parts were missing.",
    "Stopped working within the first month.",
    "The quality control is clearly nonexistent.",
    "It arrived with several scratches and dents.",
    "The sizing was completely off from the chart.",
    "Makes a strange noise that was not mentioned anywhere.",
    "It overheats after just a few minutes of use.",
    "The buttons stopped responding after a week.",
    "Color faded significantly after one wash.",
    "The zipper broke on the second use.",
    "Performance degrades noticeably over time.",
    "Packaging was poor and the item was not protected.",
    "The adhesive does not hold at all.",
    "It is much smaller than I expected from the photos.",
]

negative_closings = [
    "Do not buy this.",
    "Returning it immediately.",
    "One star is generous for this product.",
    "I wish I had read more reviews before purchasing.",
    "Complete waste of money.",
    "Would give zero stars if I could.",
    "Never buying from this brand again.",
    "Extremely poor value for money.",
    "Very unhappy with this purchase.",
]


def generate_review(sent):
    """Build a review from opening + 1-3 detail sentences + closing."""
    if sent == "positive":
        opening = np.random.choice(positive_openings)
        n_details = np.random.choice([1, 2, 3], p=[0.25, 0.50, 0.25])
        details = " ".join(np.random.choice(positive_details, n_details, replace=False))
        closing = np.random.choice(positive_closings)
    elif sent == "neutral":
        opening = np.random.choice(neutral_openings)
        n_details = np.random.choice([1, 2, 3], p=[0.30, 0.45, 0.25])
        details = " ".join(np.random.choice(neutral_details, n_details, replace=False))
        closing = np.random.choice(neutral_closings)
    else:
        opening = np.random.choice(negative_openings)
        n_details = np.random.choice([1, 2, 3], p=[0.20, 0.50, 0.30])
        details = " ".join(np.random.choice(negative_details, n_details, replace=False))
        closing = np.random.choice(negative_closings)

    return f"{opening} {details} {closing}"


# Generate all reviews
review_text = [generate_review(s) for s in sentiment]

# Compute derived features
review_length = [len(r) for r in review_text]
word_count = [len(r.split()) for r in review_text]

# Build dataframe
df = pd.DataFrame({
    "review_text": review_text,
    "sentiment": sentiment,
    "rating": rating,
    "product_category": product_category,
    "review_length": review_length,
    "word_count": word_count,
})

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "product_reviews.csv")
df.to_csv(output_path, index=False)

print(f"Saved {len(df)} reviews to {output_path}")
print(f"Columns: {list(df.columns)}")
print(f"\nSentiment distribution:")
print(df["sentiment"].value_counts().to_string())
print(f"\nRating distribution:")
print(df["rating"].value_counts().sort_index().to_string())
print(f"\nCategory distribution:")
print(df["product_category"].value_counts().to_string())
print(f"\nAvg review length: {df['review_length'].mean():.0f} chars")
print(f"Avg word count: {df['word_count'].mean():.0f} words")
print(f"\nSample review (positive):")
print(df[df["sentiment"] == "positive"]["review_text"].iloc[0][:200])
