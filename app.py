import os
import requests
import pandas as pd
import streamlit as st
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS
from google.cloud import translate_v2 as translate
import tweepy
from datetime import datetime, timedelta
import pytz
from transformers import pipeline
import torch

# ---------------- SETUP ----------------
st.set_page_config(page_title="Disaster Detection App", layout="wide")

# Google Translate setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sihproject-471916-514170694ad5.json"
translate_client = translate.Client()

# Tweepy setup
bearer_token = "AAAAAAAAAAAAAAAAAAAAAIyi4AEAAAAAvT6sMfLiUl%2FPzDe%2BiCV4NkvNzbQ%3DfoX0qsgli8JaAQLp8dcYFi6yliDVqZf2TRURigrKxhSvgnXsPB"
client = tweepy.Client(bearer_token=bearer_token)

# Hazard keywords
hazard_keywords = ["flood", "tsunami", "storm", "earthquake", "cyclone", "landslide", "fire", "oil spill", "shipwreck"]

# Zero-shot classifier setup
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)

# ---------------- EXIF + GPS Functions ----------------
def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = {}
    info = image._getexif()
    if not info:
        return None
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        if decoded == "GPSInfo":
            gps_data = {}
            for t in value:
                gps_decoded = GPSTAGS.get(t, t)
                gps_data[gps_decoded] = value[t]
            exif_data[decoded] = gps_data
        else:
            exif_data[decoded] = value
    return exif_data

def to_float(rational):
    try:
        return float(rational)
    except TypeError:
        return rational.numerator / rational.denominator

def convert_to_degrees(value):
    d, m, s = value
    return to_float(d) + to_float(m)/60 + to_float(s)/3600

def get_lat_lon(exif_data):
    if not exif_data or "GPSInfo" not in exif_data:
        return None, None
    gps_info = exif_data["GPSInfo"]
    lat = convert_to_degrees(gps_info["GPSLatitude"])
    if gps_info["GPSLatitudeRef"] != "N":
        lat = -lat
    lon = convert_to_degrees(gps_info["GPSLongitude"])
    if gps_info["GPSLongitudeRef"] != "E":
        lon = -lon
    return lat, lon

def reverse_geocode(lat, lon):
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    response = requests.get(url, headers={"User-Agent": "my-app"}).json()
    address = response.get("address", {})
    city = address.get("city") or address.get("town") or address.get("village") or None
    full_address = response.get("display_name", "Unknown")
    return city, full_address

# ---------------- Hazard & Tweets ----------------
def classify_hazard(description, target_language="en"):
    result = translate_client.translate(description, target_language=target_language)
    translated_text = result["translatedText"].lower()
    scores = {}
    for hazard in hazard_keywords:
        scores[hazard] = 1.0 if hazard in translated_text else 0.0
    best_hazard = max(scores, key=scores.get)
    return best_hazard, scores

def detect_city_from_text(text):
    result = translate_client.translate(text, target_language="en")
    translated_text = result["translatedText"].lower()
    words = translated_text.split()
    return words[0] if words else "Unknown"

def fetch_tweets(city, hazard, max_results=10):
    query = f'("{city}" OR #{city}) ("{hazard}" OR #{hazard}) -is:retweet'

    now = datetime.utcnow()
    start_time = (now - timedelta(hours=1)).isoformat("T") + "Z"
    end_time   = (now - timedelta(seconds=15)).isoformat("T") + "Z"

    local_tz = pytz.timezone("Asia/Kolkata")

    tweets = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        tweet_fields=["created_at", "text", "geo"],
        expansions=["author_id"],
        user_fields=["username"],
        start_time=start_time,
        end_time=end_time
    )

    user_map = {}
    if tweets.includes and "users" in tweets.includes:
        for u in tweets.includes["users"]:
            user_map[u["id"]] = u["username"]

    tweets_data = []
    if tweets.data:
        for t in tweets.data:
            local_time = t.created_at.astimezone(local_tz)
            formatted_time = local_time.strftime("%Y-%m-%d %H:%M:%S %Z")
            username = user_map.get(t.author_id, "Unknown")
            tweets_data.append({"time": formatted_time, "user": username, "text": t.text})
    return tweets_data

def translate_tweets_to_english(tweets_list):
    translated = []
    for tweet in tweets_list:
        result = translate_client.translate(tweet["text"], target_language="en")
        translated.append(result["translatedText"])
    return translated

def classify_tweets_disaster(translated_tweets):
    aggregate_scores = {hazard: 0.0 for hazard in hazard_keywords}
    for tweet in translated_tweets:
        res = classifier(tweet, hazard_keywords, multi_label=False)
        top_label = res['labels'][0]
        top_score = res['scores'][0]
        aggregate_scores[top_label] += top_score
    num_tweets = len(translated_tweets)
    if num_tweets == 0:
        return None, None, None
    for hazard in aggregate_scores:
        aggregate_scores[hazard] = (aggregate_scores[hazard] / num_tweets) * 100
    final_disaster = max(aggregate_scores, key=aggregate_scores.get)
    intensity = aggregate_scores[final_disaster]
    return final_disaster, intensity, aggregate_scores

# ---------------- Streamlit UI ----------------
st.title("🌍 DisasterPulse")

mode = st.radio("Choose Input Mode", ["Image + Description", "Direct Text"])

tweets_data = []
final_disaster, intensity, scores, city, full_address = None, None, None, None, None

if mode == "Direct Text":
    direct_text = st.text_area("Enter disaster-related text (any language):")
    if st.button("Analyze"):
        hazard, _ = classify_hazard(direct_text)
        city = detect_city_from_text(direct_text)
        st.write(f"**Detected City:** {city}")
        st.write(f"**Predicted Hazard:** {hazard}")
        tweets_data = fetch_tweets(city, hazard)
        translated = translate_tweets_to_english(tweets_data)
        final_disaster, intensity, scores = classify_tweets_disaster(translated)

elif mode == "Image + Description":
    uploaded_img = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    description = st.text_area("Enter disaster description (any language):")
    if uploaded_img and description and st.button("Analyze"):
        img_path = "temp_img.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_img.read())

        exif_data = get_exif_data(img_path)
        lat, lon = get_lat_lon(exif_data)
        if lat and lon:
            city, full_address = reverse_geocode(lat, lon)
            st.write(f"**Detected City:** {city}")
            st.write(f"**Full Address:** {full_address}")
        else:
            city = detect_city_from_text(description)
            st.write(f"**Detected City from text:** {city}")

        hazard, _ = classify_hazard(description)
        st.write(f"**Predicted Hazard:** {hazard}")

        tweets_data = fetch_tweets(city, hazard)
        translated = translate_tweets_to_english(tweets_data)
        final_disaster, intensity, scores = classify_tweets_disaster(translated)

# ---------------- Show Results ----------------
if final_disaster:
    st.subheader("📊 Disaster Analysis Result")
    st.write(f"**Final Disaster Type:** {final_disaster}")
    st.write(f"**Confidence (Intensity):** {intensity:.2f}%")
    st.write("### Detailed Hazard Scores:")
    st.json(scores)

    if tweets_data:
        st.subheader("📝 Fetched Tweets")
        df = pd.DataFrame(tweets_data)
        st.dataframe(df)

        # Download option
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Tweets as CSV", data=csv, file_name="tweets.csv", mime="text/csv")
    else:
        st.warning("No tweets found for this query.")
