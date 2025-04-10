import pandas as pd
import numpy as np
import os
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tempfile

# ---------- STEP 1: Parse Arts.txt ----------
def parse_arts_file(file_path):
    reviews = []
    current_review = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('product/productId:'):
                if current_review:
                    reviews.append(current_review)
                    current_review = {}
                current_review['product/productId'] = line.split(': ')[1]
            elif line.startswith('review/userId:'):
                current_review['review/userId'] = line.split(': ')[1]
            elif line.startswith('review/helpfulness:'):
                helpful = line.split(': ')[1].split('/')
                try:
                    current_review['review/helpfulness'] = [int(helpful[0]), int(helpful[1])]
                except:
                    current_review['review/helpfulness'] = [0, 1]
    if current_review:
        reviews.append(current_review)
    return reviews

# ---------- STEP 2: Helpfulness and Reliability ----------
def calculate_helpfulness_score(reviews):
    result = {}
    for review in reviews:
        user_id = review.get('review/userId')
        item_id = review.get('product/productId')
        helpful = review.get('review/helpfulness', [0, 0])
        helpful_votes = helpful[0]
        total_votes = helpful[1] if helpful[1] != 0 else 1
        bij = (helpful_votes ** 2) / total_votes
        if item_id not in result:
            result[item_id] = {}
        result[item_id][user_id] = {'bij': bij, 'review': review}
    for item_id, users in result.items():
        sum_bxj = sum(user['bij'] for user in users.values()) or 1
        for user_id in users:
            users[user_id]['hij'] = users[user_id]['bij'] / sum_bxj
    return result

def calculate_reliability_scores(helpfulness_data, alpha=0.5):
    for item_id, users in helpfulness_data.items():
        user_list = list(users.keys())
        n_prime = len(user_list)
        for i, user_id in enumerate(user_list):
            zij = sum(1 / (e**2) for e in range(1, max(n_prime - i + 1, 2)))
            sigma_squared = 1.0
            qij = (1 / sigma_squared) * (n_prime - i)
            users[user_id]['zij'] = zij
            users[user_id]['qij'] = qij
        sum_zxj = sum(users[u]['zij'] for u in user_list)
        sum_qxj = sum(users[u]['qij'] for u in user_list)
        for user_id in user_list:
            users[user_id]['mostij'] = users[user_id]['zij'] / sum_zxj
            users[user_id]['topij'] = users[user_id]['qij'] / sum_qxj
            users[user_id]['rhij'] = alpha * users[user_id]['topij'] + (1 - alpha) * users[user_id]['mostij']
            users[user_id]['avg_score'] = (users[user_id]['hij'] + users[user_id]['rhij']) / 2
    return helpfulness_data

# ---------- STEP 3: Generate DataFrame ----------
def generate_dataframe(processed_data):
    rows = []
    for item_id, users in processed_data.items():
        for user_id, data in users.items():
            rows.append({
                'user_id': user_id,
                'item_id': item_id,
                'avg_of_rhij_and_hij': data['avg_score']
            })
    return pd.DataFrame(rows)

# ---------- STEP 4: Matrix Factorization ----------
def apply_svd(df, k=50):
    pivot_df = df.pivot(index='user_id', columns='item_id', values='avg_of_rhij_and_hij').fillna(0)
    R = pivot_df.values
    U, sigma, Vt = svds(R, k=min(k, min(R.shape)-1))
    sigma = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    return pivot_df, predicted_ratings, U, Vt

# ---------- STEP 5: Evaluation ----------
def compute_loss(R_true, R_pred, U, Vt, lambda_reg=0.1):
    mask = R_true > 0
    mse_loss = np.sum((R_true[mask] - R_pred[mask]) ** 2)
    reg_term = lambda_reg * (np.sum(U**2) + np.sum(Vt**2))
    return mse_loss + reg_term

def hit_ratio_at_k(test_R, predicted_R, k=10):
    hits = 0
    total_users = 0
    for user_idx in range(test_R.shape[0]):
        true_items = np.where(test_R[user_idx] > 0)[0]
        if len(true_items) == 0:
            continue
        top_k_items = np.argsort(predicted_R[user_idx])[::-1][:k]
        if np.intersect1d(true_items, top_k_items).size > 0:
            hits += 1
        total_users += 1
    return hits / total_users if total_users else 0

# ---------- STEP 6: Main ----------
def main():
    input_path = r"C:\Users\KIIT\Desktop\mini p 8\Arts.txt"    # Change this to your file path

    try:
        reviews = parse_arts_file(input_path)
        helpfulness_data = calculate_helpfulness_score(reviews)
        processed_data = calculate_reliability_scores(helpfulness_data)
        df = generate_dataframe(processed_data)

        # Try saving CSV safely
        try:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "average_scores.csv")
            df.to_csv(output_path, index=False)
            print(f"✅ CSV saved to: {output_path}")
        except PermissionError:
            print("⚠️ Couldn't save CSV due to permission error.")

        print("\n📊 Sample of dataframe:")
        print(df.head())

        # Apply SVD
        pivot_df, predicted_ratings, U, Vt = apply_svd(df, k=50)
        R = pivot_df.values

        # Print the original rating matrix
        print("\n📦 Original Rating Matrix (R):")
        print(pd.DataFrame(R, index=pivot_df.index, columns=pivot_df.columns))

        # Print predicted matrix
        print("\n🤖 Predicted Rating Matrix (SVD output):")
        print(pd.DataFrame(predicted_ratings, index=pivot_df.index, columns=pivot_df.columns))

        # Loss
        loss = compute_loss(R, predicted_ratings, U, Vt)
        print(f"\n📉 Regularized Loss: {loss:.4f}")

        # Recommend top 10 for first user
        user_index = 0
        user_id = pivot_df.index[user_index]
        top_preds = pd.Series(predicted_ratings[user_index], index=pivot_df.columns).sort_values(ascending=False).head(10)
        print(f"\n🎯 Top 10 recommendations for user {user_id}:\n", top_preds)

        # Train-test evaluation
        user_ids = pivot_df.index.tolist()
        train_users, test_users = train_test_split(user_ids, test_size=0.2, random_state=42)
        train_matrix = pivot_df.loc[train_users].values
        test_matrix = pivot_df.loc[test_users].values

        U_train, sigma_train, Vt_train = svds(train_matrix, k=50)
        sigma_train = np.diag(sigma_train)
        test_pred = np.dot(np.dot(U_train, sigma_train), Vt_train)[:len(test_users), :]

        mask = test_matrix > 0
        mse = mean_squared_error(test_matrix[mask], test_pred[mask])
        hit_ratio = hit_ratio_at_k(test_matrix, test_pred, k=10)

        print(f"\n🧮 MSE on test set: {mse:.4f}")
        print(f"✅ Hit Ratio@10: {hit_ratio:.4f}")

    except Exception as e:
        print(f"❌ Error: {e}")

# Run the program
if __name__ == "__main__":
    main()
