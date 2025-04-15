from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import shutil
import json
import requests
import uvicorn
from openai import OpenAI
from typing import Optional, Dict, Any, List

# pydubで大容量ファイルを分割
from pydub import AudioSegment

load_dotenv()

app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-gijirokun.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 環境変数からAPIキーなど取得
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_TABLE = "minute_embeddings"


client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------
# DB保存用の Pydanticモデル
# ---------------------------------------------------------
class SaveMinutesRequest(BaseModel):
    title: str
    formatted_transcript: str
    analysis: str
    mindmap: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------
# 1) 小分割で部分要約するためのヘルパー
# ---------------------------------------------------------
def chunk_text_by_length(text: str, chunk_size: int = 3000) -> List[str]:
    """
    シンプルに文字数で chunk_size ごとに分割する関数。
    GPTのトークン上限を考慮すると3500〜4000文字ぐらいが安全。
    （今回は実質使わない形にする）
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks


def partial_summary_gpt(chunk_text: str) -> str:
    """
    1つのテキストチャンクを要約するGPT呼び出し。
    トークンオーバー回避のため max_tokens を小さめに。
    """
    prompt = f"""
あなたは会議文字起こしを要約するアシスタントです。
以下のテキストをなるべく簡潔かつ重要事項を失わないように要約してください:

{chunk_text}
"""
    print("\n[partial_summary_gpt] === CALLING GPT with prompt ===\n", prompt)
    res = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000
    )
    print("[partial_summary_gpt] === GPT RAW RESPONSE ===\n", res)
    summary_text = res.choices[0].message.content.strip()
    print("[partial_summary_gpt] === summary_text ===\n", summary_text)
    return summary_text


def combine_summaries_with_gpt(summaries: List[str]) -> str:
    """
    複数の部分要約を再度まとめて「最終要約」にするGPT呼び出し。
    """
    combined_text = "\n\n".join(summaries)
    prompt = f"""
以下は会議文字起こしの部分要約を複数集めたものです。
これらを統合し、全体の要点がわかる最終的な要約を作成してください。

{combined_text}

出力はなるべく簡潔かつ重要事項が漏れないようにしてください:
"""
    print("\n[combine_summaries_with_gpt] === CALLING GPT with prompt ===\n", prompt)
    res = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500
    )
    print("[combine_summaries_with_gpt] === GPT RAW RESPONSE ===\n", res)
    final_summary = res.choices[0].message.content.strip()
    print("[combine_summaries_with_gpt] === final_summary ===\n", final_summary)
    return final_summary


# ---------------------------------------------------------
# 2) 議事録生成ロジック
# ---------------------------------------------------------
def generate_minutes_from_text(transcript_text: str) -> dict:
    """
    1) Proofreading（整形） -> formatted_transcript
    2) （実質）全体要約で long_summary を作る
    3) GPTで最終議事録JSON生成
    """

    print("==== generate_minutes_from_text ====")
    print("[generate_minutes_from_text] 【入力全文】:\n", transcript_text)

    # (A) 文字起こしの整形 (Proofreading)
    proofreading_prompt = f"""
            1. 文字起こしの整形プロンプト

            目的：話者ごとの区切りを保ちつつ、内容は変えずに、口語の無駄（例：「えーと」「あのー」「うーん」など）を省いて読みやすく整形する

            # あなたの役割：
            あなたは音声文字起こしの整形担当です。

            # タスクの説明：
            以下に提供される会話形式の文字起こしデータには、「えーと」「あのー」「うーん」などの無駄なつなぎ言葉や言い淀み、繰り返し、話が逸れる部分が含まれています。
            この内容を読みやすく整形してください。ただし、発言者ごとの順序や内容自体は絶対に変更せず、意味が変わらないようにしてください。

            # 出力ルール：
            - 話者ごとに段落を分け、話し言葉のままに近い自然な表現に整えてください
            - 「えーと」「あのー」「うーん」などの無駄な言葉は削除してください
            - 繰り返しの言葉や話の脱線も、省ける範囲で整えてください
            - 敬語や文末は元の雰囲気をなるべく保ってください

            # 出力形式：
            話者名：発言内容
            （改行）

            ---
            # 入力データ：
            {transcript_text}
            """
    print("\n[generate_minutes_from_text] === CALLING GPT for Proofreading Prompt ===\n", proofreading_prompt)
    proofreading_res = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": proofreading_prompt}],
        temperature=0.1,
        max_tokens=2000
    )
    print("[generate_minutes_from_text] === GPT RAW RESPONSE (Proofreading) ===\n", proofreading_res)
    formatted_transcript = proofreading_res.choices[0].message.content.strip()
    print("[generate_minutes_from_text] === formatted_transcript ===\n", formatted_transcript)

    # (B) 長文分割せずに一括処理
    if len(formatted_transcript) > 9999999:
        chunked_texts = chunk_text_by_length(formatted_transcript, 3000)
        partials = []
        for i, chunk in enumerate(chunked_texts):
            print(f"[generate_minutes_from_text] Partial summary chunk {i+1}/{len(chunked_texts)}")
            partial_sum = partial_summary_gpt(chunk)
            partials.append(partial_sum)
        long_summary = combine_summaries_with_gpt(partials)
    else:
        long_summary = formatted_transcript

    print("\n[generate_minutes_from_text] === long_summary (after no-chunk or partial-chunk) ===\n", long_summary)

    # (C) Embedding & 過去議事録検索
    print("\n[generate_minutes_from_text] === Creating Embedding ===")
    emb_res = client.embeddings.create(
        input=[formatted_transcript],
        model="text-embedding-ada-002"
    )
    query_embedding = emb_res.data[0].embedding
    print("[generate_minutes_from_text] === Embedding Created ===\n", query_embedding)

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
    print("[generate_minutes_from_text] === Searching Past Minutes with Payload ===\n", rpc_payload)
    rpc_resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/match_minutes",
        headers=headers,
        json=rpc_payload
    )
    matched_minutes = rpc_resp.json()
    print("[generate_minutes_from_text] === matched_minutes ===\n", matched_minutes)

    if isinstance(matched_minutes, list) and len(matched_minutes) > 0:
        if isinstance(matched_minutes[0], dict):
            past_analysis_texts = "\n\n".join(item.get("analysis", "") for item in matched_minutes)
        else:
            past_analysis_texts = ""
    else:
        past_analysis_texts = ""

    # (D) GPTで最終的な議事録JSON生成
    analysis_prompt = f"""
    あなたは企業の戦略会議を専門とする高度な議事録作成アシスタントです。

    【過去の議事録（参照用）】
    {past_analysis_texts}

    【今回の会議内容 (要約/整形後)】
    {long_summary}

　　　議事録の文章作成は以下の項目に沿った記述を必ず守り、会議の内容を正確に反映し、後で振り返る際に役立つようにしてください。
    上記を踏まえて、 以下の項目をそれぞれ「具体的に」「最低でも3行以上」でまとめてください。
    
    
    ■タイトル（この会議を一言で表すキャッチーなタイトル）
    ■会議サマリー（全体概要を5〜7行以上）
    ■決定事項（明確に箇条書きで、最低3項目以上）
    ■課題・懸念点（未解決事項を箇条書きで最低3項目以上）
    ■今後の方針・アクションプラン（具体的に担当者と期限を明記し、最低3項目以上）
    ■要因分析（課題や成功/失敗要因を明確に分析し、なるべく網羅的に最低3項目以上）
    ■事実と解釈の仕分け（事実情報と主観的解釈を分離して整理し、最低3項目以上）

　　　改善案の例文を以下に示します。
    【例文】
        本会議では、AI議事録アシスタントの市場導入戦略が議論されました。開発完了後の次のステップとして、ターゲット市場の特定と効果的な広告戦略の策定が中心となりました。具体的な広告コピーとランディングページのリライト、動画コンテンツの制作が計画されています。
        決定事項：
        1. 広告コピーのABテストを実施。
        2. ランディングページのリライト。
        3. 動画コンテンツの制作。
        課題・懸念点：
        - ターゲット市場のさらなる絞り込み。
        - 効果的な広告戦略の確立。
        今後の方針・アクションプラン：
        - 広告コピーのABテストを5月中に実施（鈴木担当）。
        - ランディングページのリライトを6月初旬までに完了（山本担当）。
        - 動画コンテンツの制作を6月末までに完了（佐藤担当）。
        要因分析：
        - 成功要因は明確なターゲット市場の特定と効果的な広告戦略。
        - 課題は市場のニーズに合わせた広告コピーの開発。
        事実と解釈の仕分け：
        事実：AI議事録アシスタントの開発は完了している。
        解釈：市場導入の成功は広告戦略とターゲット市場の適切な特定に依存する。

    
    出力は以下のJSON形式のみで返してください。
    
    
    childrenの数は適宜調整してください：
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
                        {{
                            "name": "決定事項",
                            "children": [{{"name": "決定事項1"}}, {{"name": "決定事項2"}}]
                        }}
                        {{
                            "name": "課題・懸念点",
                            "children": [{{"name": "課題1"}}, {{"name": "課題2"}}]
                        }}
                        {{
                            "name": "今後の方針",
                            "children": [{{"name": "方針1"}}, {{"name": "方針2"}}]
                        }}
                        {{
                            "name": "要因分析",
                            "children": [{{"name": "要因1"}}, {{"name": "要因2"}}]
                        }}
                        {{
                            "name": "事実と解釈の仕分け",
                            "children": [{{"name": "事実1"}}, {{"name": "解釈2"}}]
                        }}
                    ]
                }}
            ]
        }}
    }}
    """
    print("\n[generate_minutes_from_text] === CALLING GPT for Final JSON ===\n", analysis_prompt)
    final_res = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": analysis_prompt}],
        temperature=0.2,
        max_tokens=3500
    )
    print("[generate_minutes_from_text] === GPT RAW RESPONSE (Analysis) ===\n", final_res)
    analysis_raw = final_res.choices[0].message.content.strip()
    print("[generate_minutes_from_text] === analysis_raw (Full GPT Output) ===\n", analysis_raw)

    try:
        analysis_json = json.loads(analysis_raw)
        print("[generate_minutes_from_text] === Successfully parsed JSON ===\n", analysis_json)
    except json.JSONDecodeError:
        print("[generate_minutes_from_text] !!! JSONDecodeError. Using fallback structure.")
        analysis_json = {
            "タイトル": "不明",
            "議事録": analysis_raw,
            "マインドマップ": None
        }

    output_dict = {
        "formatted_transcript": formatted_transcript,
        "title": analysis_json.get("タイトル", "タイトル不明"),
        "analysis": analysis_json.get("議事録", ""),
        "mindmap": analysis_json.get("マインドマップ", None)
    }

    print("\n[generate_minutes_from_text] === Final Output ===\n", output_dict)
    return output_dict


# ---------------------------------------------------------
# /transcribe - 25MB超なら分割, Whisper language="ja"
# ---------------------------------------------------------
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    React から音声ファイルを受け取る。
    25MB超の場合は pydub で分割→Whisper。
    取得した全文文字起こしを generate_minutes_from_text() へ。
    """
    temp_path = "./temp_audio.webm"
    try:
        print("\n[/transcribe] === Received file ===", audio.filename)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        file_size = os.path.getsize(temp_path)
        print(f"[/transcribe] 受信ファイルサイズ: {file_size} bytes")

        transcript = ""

        # Whisperは response_format="text" -> 戻り値は str
        # language="ja"指定で日本語認識精度アップを期待
        if file_size <= 25 * 1024 * 1024:
            print("[/transcribe] => 25MB以下: Whisperを1回だけ実行")
            with open(temp_path, "rb") as f_in:
                transcript_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f_in,
                    response_format="text",
                    language="ja"
                )
            print("[/transcribe] Whisper response:\n", transcript_response)
            transcript = transcript_response
        else:
            print("[/transcribe] => 25MB超: pydubで分割し、複数回Whisper実行")
            chunk_ms = 10 * 60 * 1000
            audio_segment = AudioSegment.from_file(temp_path)

            start_ms = 0
            idx = 0
            while start_ms < len(audio_segment):
                end_ms = start_ms + chunk_ms
                chunk = audio_segment[start_ms:end_ms]
                chunk_path = f"./temp_chunk_{idx}.mp3"
                print(f"[/transcribe] => chunk export idx={idx}, {start_ms}~{end_ms}ms => {chunk_path}")
                chunk.export(chunk_path, format="mp3", bitrate="64k")

                with open(chunk_path, "rb") as f_chunk:
                    chunk_res = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f_chunk,
                        response_format="text",
                        language="ja"
                    )
                print(f"[/transcribe] => chunk {idx} Whisper response:\n", chunk_res)
                transcript += chunk_res + "\n"
                os.remove(chunk_path)
                idx += 1
                start_ms = end_ms

        # 生成ロジック
        result = generate_minutes_from_text(transcript)
        print("[/transcribe] === Final Result ===\n", result)
        return result

    except Exception as e:
        print("[/transcribe] エラー:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------
# /transcribe-text (テキストモード)
# ---------------------------------------------------------
@app.post("/transcribe-text")
async def transcribe_text(payload: dict = Body(...)):
    raw_text = payload.get("raw_text", "")
    print("\n[/transcribe-text] Raw input:\n", raw_text)
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="テキストが空です")
    result = generate_minutes_from_text(raw_text)
    print("[/transcribe-text] => Final Result:\n", result)
    return result


# ---------------------------------------------------------
# /transcribe-chunks (複数ファイル)
# ---------------------------------------------------------
@app.post("/transcribe-chunks")
async def transcribe_chunks(audios: list[UploadFile] = File(...)):
    combined_transcript = ""
    print("\n[/transcribe-chunks] => Received multiple audio files, count:", len(audios))
    for idx, audio in enumerate(audios):
        print(f"[/transcribe-chunks] => Handling file {idx}: {audio.filename}")
        ext = os.path.splitext(audio.filename)[1]
        if not ext:
            ext = ".webm"
        temp_path = f"./temp_audio_chunk_{idx}{ext}"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        with open(temp_path, "rb") as f_in:
            chunk_res = client.audio.transcriptions.create(
                model="whisper-1",
                file=f_in,
                response_format="text",
                language="ja"
            )
        print(f"[/transcribe-chunks] => chunk {idx} Whisper response:\n", chunk_res)
        combined_transcript += chunk_res + "\n"
        os.remove(temp_path)

    if not combined_transcript.strip():
        raise HTTPException(status_code=400, detail="音声チャンクの文字起こしに失敗しました")

    result = generate_minutes_from_text(combined_transcript)
    print("[/transcribe-chunks] => Final Result:\n", result)
    return result


# ---------------------------------------------------------
# /save-minutes (議事録保存)
# ---------------------------------------------------------
@app.post("/save-minutes")
async def save_minutes(data: SaveMinutesRequest):
    print("\n[/save-minutes] => Incoming data:\n", data)
    if isinstance(data.mindmap, str):
        try:
            data.mindmap = json.loads(data.mindmap)
        except:
            data.mindmap = None

    print("[/save-minutes] => mindmap after parse:\n", data.mindmap)

    emb_res = client.embeddings.create(
        input=[data.formatted_transcript],
        model="text-embedding-ada-002"
    )
    embedding_vector = emb_res.data[0].embedding
    print("[/save-minutes] => embedding_vector:\n", embedding_vector)

    payload = {
        "title": data.title,
        "formatted_transcript": data.formatted_transcript,
        "analysis": data.analysis,
        "mindmap": data.mindmap,
        "embedding": embedding_vector
    }
    print("[/save-minutes] => Inserting to DB payload:\n", payload)

    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
        headers=headers,
        json=payload
    )
    if r.status_code in [200, 201, 204]:
        print("[/save-minutes] => Save success:", r.text)
        return {"status": "success"}
    else:
        print("[/save-minutes] => Save error:", r.text, r.status_code)
        return {
            "status": "error",
            "detail": r.text,
            "code": r.status_code
        }


# ---------------------------------------------------------
# /get-minutes (議事録一覧)
# ---------------------------------------------------------
@app.get("/get-minutes")
async def get_minutes():
    print("\n[/get-minutes] => fetching from DB...")
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
    }
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?select=*",
        headers=headers
    )
    if r.status_code in [200, 201, 204]:
        print("[/get-minutes] => success")
        return {"minutes": r.json()}
    else:
        print("[/get-minutes] => error", r.text, r.status_code)
        return {
            "status": "error",
            "detail": r.text,
            "code": r.status_code
        }


# ---------------------------------------------------------
# /delete-minutes/{minute_id} (削除)
# ---------------------------------------------------------
@app.delete("/delete-minutes/{minute_id}")
async def delete_minutes(minute_id: str):
    print(f"\n[/delete-minutes] => minute_id={minute_id}")
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?id=eq.{minute_id}"
    r = requests.delete(url, headers=headers)
    if r.status_code in [200, 204]:
        print("[/delete-minutes] => Delete success")
        return {"status": "success"}
    else:
        print("[/delete-minutes] => Delete error:", r.text, r.status_code)
        return {
            "status": "error",
            "detail": r.text,
            "code": r.status_code
        }


# ---------------------------------------------------------
# /chatbot (チャットモード)
# ---------------------------------------------------------
@app.post("/chatbot")
async def chatbot_query(payload: dict = Body(...)):
    user_message = payload.get("message", "")
    print("\n[/chatbot] => user_message:\n", user_message)
    if not user_message.strip():
        raise HTTPException(status_code=400, detail="メッセージが空です")

    user_emb = client.embeddings.create(
        input=[user_message],
        model="text-embedding-ada-002"
    )
    user_query_vector = user_emb.data[0].embedding
    print("[/chatbot] => user_query_vector:\n", user_query_vector)

    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json"
    }
    rpc_payload = {
        "query_embedding": user_query_vector,
        "match_threshold": 0.2,
        "match_count": 5
    }
    print("[/chatbot] => Searching Past Minutes with Payload:\n", rpc_payload)
    rpc_resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/match_minutes",
        headers=headers,
        json=rpc_payload
    )
    matched_minutes = rpc_resp.json()
    print("[/chatbot] => matched_minutes:\n", matched_minutes)

    if isinstance(matched_minutes, list) and len(matched_minutes) > 0:
        if isinstance(matched_minutes[0], dict):
            retrieved_texts = "\n\n".join(item.get("analysis", "") for item in matched_minutes)
        else:
            retrieved_texts = ""
    else:
        retrieved_texts = ""

    system_prompt = """
    あなたは社内議事録のデータベースを参照できるAIアシスタントです。
    ユーザーの質問に対して、議事録データに基づき、できるだけ具体的に回答してください。
    必要に応じて推測も構いませんが、事実と推測は分けて表現してください。
    """
    user_prompt = f"""
    【過去の議事録抜粋】
    {retrieved_texts}

    【ユーザーからの質問】
    {user_message}

    上記情報を踏まえて、簡潔かつ具体的に回答してください:
    """

    print("\n[/chatbot] => GPT system_prompt:\n", system_prompt)
    print("[/chatbot] => GPT user_prompt:\n", user_prompt)

    gpt_res = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=800
    )
    print("[/chatbot] => GPT RAW RESPONSE:\n", gpt_res)
    answer_text = gpt_res.choices[0].message.content.strip()

    print("[/chatbot] => Final answer_text:\n", answer_text)
    return {"response": answer_text}


# ---------------------------------------------------------
# メイン起動
# ---------------------------------------------------------
if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
