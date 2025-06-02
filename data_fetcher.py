"""
This module contains the DataFetcher class, which is used to fetch data from
MedlinePlus API for popular health topics and their associated drug-related information.
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json
import time


class DataFetcher:
    BASE_URL = "https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term="

    def __init__(self, topics: List[str]):
        self.topics = topics

    def strip_html(self, html: str) -> str:
        """Clean HTML content by removing tags and normalizing whitespace."""
        if not html:
            return ""
        # Create a new BeautifulSoup object to handle the HTML content
        soup = BeautifulSoup(html, "html.parser")
        # Remove any script or style elements
        for element in soup(["script", "style"]):
            element.decompose()
        # Get text and normalize whitespace
        text = soup.get_text(separator=" ", strip=True)
        return text

    def fetch_articles(self, query: str) -> List[Dict]:
        encoded_query = requests.utils.quote(query)
        url = f"{self.BASE_URL}{encoded_query}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f" Failed to fetch data for query: {query}")
            return []

        soup = BeautifulSoup(response.content, "lxml")
        articles = []

        for doc in soup.find_all("document"):
            try:
                title_tag = doc.find("content", {"name": "title"})
                snippet_tag = doc.find("content", {"name": "snippet"})
                full_tag = doc.find("content", {"name": "FullSummary"})
                url_attr = doc.get("url")

                if title_tag and url_attr:
                    # Convert the entire content to string and then clean it
                    title_html = ''.join(str(item) for item in title_tag.contents)
                    snippet_html = ''.join(str(item) for item in snippet_tag.contents) if snippet_tag else ""
                    full_html = ''.join(str(item) for item in full_tag.contents) if full_tag else ""

                    articles.append({
                        "title": self.strip_html(title_html),
                        "snippet": self.strip_html(snippet_html),
                        "full_text": self.strip_html(full_html),
                        "url": url_attr
                    })

            except Exception as e:
                print(f" Failed to parse document: {e}")
                continue

        return articles

    def fetch_topic_data(self) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Fetches both general and drug-related articles for each topic.
        Returns a dictionary of all results.
        """
        all_data = {}

        for topic in self.topics:
            print(f"🔍 Fetching: {topic}")
            try:
                health_query = f'"{topic}"'
                drug_query = f'"{topic} medicines" OR "{topic} drugs"'

                health_articles = self.fetch_articles(health_query)
                drug_articles = self.fetch_articles(drug_query)

                all_data[topic] = {
                    "health_articles": health_articles,
                    "drug_articles": drug_articles
                }

                time.sleep(0.5)  # polite delay

            except Exception as e:
                print(f" Error fetching data for {topic}: {e}")

        return all_data


if __name__ == "__main__":
    top_topics = [
        "Diabetes", "Asthma", "Hypertension", "Depression", "Anxiety", "Heart Disease",
        "Arthritis", "Obesity", "High Cholesterol", "Cancer", "COVID-19", "Flu (Influenza)",
        "Pneumonia", "Stroke", "Migraine", "Alzheimer's Disease", "Parkinson's Disease",
        "Chronic Pain", "Acid Reflux (GERD)", "Back Pain", "COPD", "Sleep Disorders",
        "Allergies", "Autism", "Bipolar Disorder", "Breast Cancer", "Lung Cancer",
        "Colon Cancer", "Prostate Cancer", "Skin Cancer", "UTI (Urinary Tract Infection)",
        "Osteoporosis", "HIV/AIDS", "STDs", "ADHD", "Epilepsy", "Kidney Disease",
        "Liver Disease", "Gallstones", "Appendicitis", "Celiac Disease", "Crohn's Disease",
        "Ulcerative Colitis", "Diverticulitis", "Pancreatitis", "Hepatitis", "Tuberculosis",
        "Sickle Cell Disease", "Anemia", "Thyroid Disorders", "Menopause", "Infertility",
        "PCOS", "Endometriosis", "Pregnancy", "Prenatal Care", "Child Development",
        "Vaccines", "Mental Health", "Dental Health", "Vision Problems", "Hearing Loss",
        "Vertigo", "Sinusitis", "Tonsillitis", "Ear Infections", "Skin Conditions",
        "Eczema", "Psoriasis", "Acne", "Warts", "Shingles", "Lupus", "Multiple Sclerosis",
        "ALS", "Dementia", "Eating Disorders", "Bulimia", "Anorexia", "Substance Abuse",
        "Alcohol Use Disorder", "Smoking Cessation", "Pain Management", "First Aid",
        "Injuries", "Burns", "Fractures", "Sprains", "Exercise and Fitness", "Nutrition",
        "Healthy Eating", "Weight Loss", "Childhood Obesity", "Men's Health", "Women's Health",
        "Aging", "Grief", "Caregiving", "Medical Tests", "Blood Pressure", "Covid-19"
    ]

    fetcher = DataFetcher(top_topics)
    results = fetcher.fetch_topic_data()

    with open("topic_article_store.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(" All cleaned topic data saved to 'topic_article_store.json'")
