import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies = pd.DataFrame({
    'title': ['Inception', 'The Matrix', 'Interstellar', 'The Prestige', 'The Dark Knight'],
    'description': [
        'A thief who enters dreams to steal secrets is given a mission to plant an idea.',
        'A hacker discovers reality is a simulation and fights against its controllers.',
        'Explorers travel through a wormhole in space to ensure humanity’s survival.',
        'Two magicians compete, leading to obsession, deceit, and deadly consequences.',
        'Batman faces the Joker, who seeks to create chaos in Gotham City.'
    ]
})
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
def recommend_movies(movie_title, num_recommendations=3):
    if movie_title not in movies['title'].values:
        return "Movie not found in the dataset."

    movie_idx = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_movie_indices = [idx for idx, score in similarity_scores[1:num_recommendations+1]]
    return movies['title'].iloc[recommended_movie_indices].tolist()
print(recommend_movies('Inception'))
