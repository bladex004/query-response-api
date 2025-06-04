from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os

app = FastAPI(title="AI-Driven Query Response API")

class QueryRequest(BaseModel):
    query: str

class CustomAI:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
        self.training_data = {"queries": [], "cluster_labels": []}
        self.data_file = "training_data.json"
        self.query_count = 0
        self.train_interval = 10
        self.used_response_sets = {}  # Query -> set of response sets (tuples)
        self.num_clusters = 10
        self.response_formats = [
            "To {query}, please use the website’s designated process.",
            "For {query}, follow the instructions on our site.",
            "Regarding {query}, check the appropriate section on the website.",
            "To address {query}, refer to the site’s guidelines.",
            "For assistance with {query}, explore the relevant options online.",
            "To handle {query}, use the website’s provided tools.",
            "Concerning {query}, navigate to the site’s resources."
        ]
        self.load_training_data()

    def load_training_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r") as f:
                    self.training_data = json.load(f)
            except Exception as e:
                print(f"Error loading training data: {e}")

    def save_training_data(self):
        try:
            with open(self.data_file, "w") as f:
                json.dump(self.training_data, f)
        except Exception as e:
            print(f"Error saving training data: {e}")

    def custom_cluster(self, features):
        if features.shape[0] < self.num_clusters:
            return [-1] * features.shape[0]
        indices = random.sample(range(features.shape[0]), self.num_clusters)
        centroids = np.array([features[i].toarray()[0] for i in indices])
        labels = [-1] * features.shape[0]
        max_iterations = 10
        for _ in range(max_iterations):
            new_labels = []
            for i in range(features.shape[0]):
                distances = np.linalg.norm(features[i].toarray() - centroids, axis=1)
                new_labels.append(np.argmin(distances))
            if new_labels == labels:
                break
            labels = new_labels[:]
            for k in range(self.num_clusters):
                cluster_points = [features[i].toarray()[0]
                                 for i in range(features.shape[0]) if labels[i] == k]
                if cluster_points:
                    centroids[k] = np.mean(cluster_points, axis=0)
        return labels

    def train_model(self, query):
        self.query_count += 1
        self.training_data["queries"].append(query)
        self.training_data["cluster_labels"].append(-1)

        if self.query_count % self.train_interval == 0 and len(self.training_data["queries"]) >= self.num_clusters:
            try:
                features = self.vectorizer.fit_transform(self.training_data["queries"])
                self.training_data["cluster_labels"] = self.custom_cluster(features)
                self.save_training_data()
                return "Model updated automatically."
            except Exception as e:
                return f"Error updating model: {e}"
        return None

    def generate_initial_responses(self, query):
        available_formats = [
            f for f in self.response_formats
            if f.format(query=query.lower()) not in {r for rs in self.used_response_sets.get(query, []) for r in rs}
        ]
        if len(available_formats) < 3:
            self.used_response_sets[query] = []
            available_formats = self.response_formats[:]
        selected_formats = random.sample(available_formats, min(3, len(available_formats)))
        responses = [f.format(query=query.lower()) for f in selected_formats]
        return responses

    def generate_responses(self, query):
        if query not in self.used_response_sets:
            self.used_response_sets[query] = []

        update_status = self.train_model(query)
        if len(self.training_data["queries"]) < self.num_clusters:
            responses = self.generate_initial_responses(query)
            status = "Success"
        else:
            try:
                query_features = self.vectorizer.transform([query])
                features = self.vectorizer.transform(self.training_data["queries"])
                distances = [np.linalg.norm(query_features.toarray()[0] - f.toarray()[0]) for f in features]
                cluster_indices = [
                    i for i, label in enumerate(self.training_data["cluster_labels"])
                    if label == self.training_data["cluster_labels"][np.argmin(distances)]
                ]
                if cluster_indices:
                    available_queries = [
                        self.training_data["queries"][i] for i in cluster_indices
                        if all(self.response_formats[0].format(query=self.training_data["queries"][i].lower()) not in rs
                               for rs in self.used_response_sets[query])
                    ]
                    if len(available_queries) < 3:
                        self.used_response_sets[query] = []
                        available_queries = [self.training_data["queries"][i] for i in cluster_indices]
                    num_responses = min(3, len(available_queries))
                    selected_queries = random.sample(available_queries, num_responses)
                    responses = []
                    used_formats = set()
                    for q in selected_queries:
                        available_formats = [
                            f for f in self.response_formats
                            if f not in used_formats and f.format(query=q.lower()) not in {r for rs in self.used_response_sets[query] for r in rs}
                        ]
                        if not available_formats:
                            available_formats = [f for f in self.response_formats if f not in used_formats]
                        if available_formats:
                            fmt = random.choice(available_formats)
                            responses.append(fmt.format(query=q.lower()))
                            used_formats.add(fmt)
                        else:
                            responses.append(self.response_formats[0].format(query=q.lower()))
                    status = "Success"
                else:
                    responses = self.generate_initial_responses(query)
                    status = "Success"
            except Exception as e:
                responses = self.generate_initial_responses(query)
                status = f"Model error: {e}"

        self.used_response_sets[query].append(tuple(responses))
        return responses, status or update_status or "Success"

# Initialize CustomAI instance
ai = CustomAI()

@app.post("/querydesk/generate-responses")
async def generate_responses(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    responses, status = ai.generate_responses(query)
    return {
        "query": query,
        "responses": responses,
        "status": status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)