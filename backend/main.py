from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI
import shutil
import json
import requests

load_dotenv()

app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-gijirokun.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 環境変数からAPIキーなど取得
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_TABLE = "minute_embeddings"

class SaveMinutesRequest(BaseModel):
    formatted_transcript: str
    analysis: str
    mindmap: dict
    title: str

@app.post("/transcribe")
async def transcribe_and_analyze(audio: UploadFile = File(...)):
    audio_path = "./temp_audio.webm"
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    with open(audio_path, "rb") as audio_file:
        transcript_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    transcript = transcript_response.text

    proofreading_prompt = f"""
    あなたは文字起こしの整形専門アシスタントです。
    以下を文法的に整理し、読みやすく整えてください。
    ・句読点や改行を適切に入れる
    ・冗長な口語表現を削除
    ・ニュアンスや重要な部分を維持

    【文章】
    {transcript}
    """

    proofreading_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": proofreading_prompt}],
        temperature=0.1,
        max_tokens=1500
    )
    formatted_transcript = proofreading_response.choices[0].message.content

    embedding_response = client.embeddings.create(
        input=[formatted_transcript],
        model="text-embedding-ada-002"
    )
    query_embedding = embedding_response.data[0].embedding

    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json"
    }
    rpc_payload = {
        "query_embedding": query_embedding,
        "match_threshold": 0.2,
        "match_count": 5
    }
    rpc_response = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/match_minutes",
        headers=headers,
        json=rpc_payload
    )
    matched_minutes = rpc_response.json()

    if isinstance(matched_minutes, list):
        if matched_minutes and isinstance(matched_minutes[0], dict):
            past_analysis_texts = "\n\n".join(minute.get("analysis", "") for minute in matched_minutes)
        elif matched_minutes and isinstance(matched_minutes[0], str):
            past_analysis_texts = "\n\n".join(matched_minutes)
        else:
            past_analysis_texts = ""
    else:
        print("⚠️ matched_minutes の中身:", matched_minutes)
        raise ValueError("SupabaseのRPC match_minutes からの戻り値が想定外の形式です。")

    analysis_prompt = f"""
    あなたは企業の戦略会議を専門とする高度な議事録作成アシスタントです。

    【過去の議事録（参照用）】
    {past_analysis_texts}

    【今回の会議内容】
    {formatted_transcript}

    上記を踏まえて、以下の項目を詳細にまとめてください。

    ■タイトル（この会議を一言で表すキャッチーなタイトル）
    ■会議サマリー（全体概要を3〜5行程度）
    ■決定事項（明確に箇条書き）
    ■課題・懸念点（未解決事項を箇条書き）
    ■今後の方針・アクションプラン（具体的に担当者と期限を明記）
    ■要因分析（課題や成功/失敗要因を明確に分析）
    ■事実と解釈の仕分け（事実情報と主観的解釈を分離して整理）

    出力は以下のJSON形式のみで返してください：

    {{
        "タイトル": "この会議のタイトル",
        "議事録": "サマリー、決定事項、課題・懸念点、今後の方針・アクションプラン、要因分析、事実と解釈の仕分けを含む本文",
        "改善案": "過去データを踏まえた具体的改善案",
        "マインドマップ": {{
            "name": "会議",
            "children": [
                {{
                    "name": "議題",
                    "children": [
                        {{
                            "name": "課題",
                            "children": [{{"name": "原因・背景"}}, {{"name": "原因2"}}]
                        }},
                        {{
                            "name": "改善案",
                            "children": [{{"name": "提案1"}}, {{"name": "提案2"}}]
                        }}
                    ]
                }}
            ]
        }}
    }}
    """

    analysis_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": analysis_prompt}],
        temperature=0.2,
        max_tokens=2000
    )
    analysis_raw = analysis_response.choices[0].message.content.strip()

    try:
        analysis_json = json.loads(analysis_raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {e}\n{analysis_raw}")

    return {
        "formatted_transcript": formatted_transcript,
        "title": analysis_json["タイトル"],
        "analysis": analysis_json["議事録"],
        "improvement": analysis_json["改善案"],
        "mindmap": analysis_json["マインドマップ"]
    }

@app.post("/save-minutes")
async def save_minutes(data: SaveMinutesRequest):
    embedding_response = client.embeddings.create(
        input=[data.formatted_transcript],
        model="text-embedding-ada-002"
    )
    embedding_vector = embedding_response.data[0].embedding
    payload = data.dict()
    payload["embedding"] = embedding_vector

    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
        headers=headers,
        json=payload
    )
    if response.status_code in [200, 201, 204]:
        return {"status": "success"}
    else:
        return {"status": "error", "detail": response.text, "code": response.status_code}

@app.get("/get-minutes")
async def get_minutes():
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
    }
    response = requests.get(
        f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?select=*",
        headers=headers
    )
    if response.status_code in [200, 201, 204]:
        return {"minutes": response.json()}
    else:
        return {"status": "error", "detail": response.text, "code": response.status_code}
