import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Load dataset
ratings = pd.read_csv('data/ml-latest-small/ratings.csv')  # userId, movieId, rating, timestamp
movies = pd.read_csv('data/ml-latest-small/movies.csv')  # movieId, title, genres

# Merge ratings and movie titles
data = pd.merge(ratings, movies, on='movieId')

# Create user-item interaction matrix
user_item_matrix = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
user_item_matrix_np = user_item_matrix.values

# Define Matrix Factorization Model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)

    def forward(self, user, item):
        user_embedding = self.user_factors(user)
        item_embedding = self.item_factors(item)
        return (user_embedding * item_embedding).sum(1)

# Model initialization
num_users, num_items = user_item_matrix.shape
num_factors = 20  # Latent factors
model = MatrixFactorization(num_users, num_items, num_factors)

# Prepare training data
user_item_pairs = np.array([(i, j, user_item_matrix_np[i, j]) for i in range(num_users) for j in range(num_items) if user_item_matrix_np[i, j] > 0])
train_data, test_data = train_test_split(user_item_pairs, test_size=0.2)

train_users = torch.LongTensor(train_data[:, 0])
train_items = torch.LongTensor(train_data[:, 1])
train_ratings = torch.FloatTensor(train_data[:, 2])

test_users = torch.LongTensor(test_data[:, 0])
test_items = torch.LongTensor(test_data[:, 1])
test_ratings = torch.FloatTensor(test_data[:, 2])

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 20

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    predictions = model(train_users, train_items)
    loss = criterion(predictions, train_ratings)
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluation
model.eval()
with torch.no_grad():
    test_predictions = model(test_users, test_items)
    test_loss = criterion(test_predictions, test_ratings)
    
print(f'Test Loss: {test_loss.item()}')

# Movie Recommendation Function
def recommend_movies(user_id, model, user_item_matrix, num_recommendations=5):
    user_idx = user_id - 1  # Adjust for zero-indexing
    user_ratings = user_item_matrix[user_idx, :]
    unrated_movies = np.where(user_ratings == 0)[0]
    
    user_tensor = torch.LongTensor([user_idx] * len(unrated_movies))
    item_tensor = torch.LongTensor(unrated_movies)
    
    model.eval()
    with torch.no_grad():
        predictions = model(user_tensor, item_tensor)
        
    top_movie_idxs = predictions.argsort(descending=True)[:num_recommendations]
    recommended_movie_ids = unrated_movies[top_movie_idxs]
    
    return movies[movies['movieId'].isin(recommended_movie_ids)]

# Example: Get top 5 movie recommendations for user 1
recommended_movies = recommend_movies(1, model, user_item_matrix_np)
print(recommended_movies[['title']])
