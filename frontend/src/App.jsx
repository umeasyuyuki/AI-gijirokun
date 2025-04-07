import React, { useState, useEffect } from 'react';
import Tree from 'react-d3-tree';
import { ReactMediaRecorder } from 'react-media-recorder';

function App() {
  // --------------------------
  // ステート定義
  // --------------------------
  const [notes, setNotes] = useState('ここに整形された文字起こしが表示されます');
  const [summary, setSummary] = useState('ここに議事録・改善案が表示されます');
  const [mindmapData, setMindmapData] = useState(null);
  const [title, setTitle] = useState('');

  const [recording, setRecording] = useState(false);
  const [recorded, setRecorded] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);

  const [loading, setLoading] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null);

  const [savedMinutes, setSavedMinutes] = useState([]);
  const [selectedMinute, setSelectedMinute] = useState(null);

  const [sidebarOpen, setSidebarOpen] = useState(true);

  // --------------------------
  // APIから保存済み議事録を取得
  // --------------------------
  const fetchSavedMinutes = async () => {
    try {
     const res = await fetch(`${process.env.REACT_APP_API_URL}/get-minutes`);
      if (!res.ok) throw new Error('保存された議事録の取得に失敗しました');
      const data = await res.json();
      // 直近のものが上になるようソート（id降順）
      const sorted = data.minutes.sort((a, b) => b.id - a.id);
      setSavedMinutes(sorted);
    } catch (error) {
      console.error('Error fetching saved minutes:', error);
    }
  };

  useEffect(() => {
    fetchSavedMinutes();
  }, []);

  // --------------------------
  // 録音データをサーバーに送信して生成結果を取得
  // --------------------------
  const sendAudioToServer = async (audioBlob) => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    try {
      setLoading(true);
      const res = await fetch(`${process.env.REACT_APP_API_URL}/transcribe`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`サーバーエラー: ${res.statusText}`);
      const data = await res.json();
      // 取得結果は編集可能な状態で各ステートにセット
      setNotes(data.formatted_transcript);
      setSummary(data.analysis);
      setMindmapData(data.mindmap);
      setTitle(data.title); // GPTが生成したタイトル
    } catch (error) {
      console.error('通信エラー:', error);
      setNotes('通信エラーが発生しました');
    } finally {
      setLoading(false);
    }
  };

  // --------------------------
  // 議事録保存
  // --------------------------
  const saveToDatabase = async () => {
    try {
     const res = await fetch(`${process.env.REACT_APP_API_URL}/save-minutes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          formatted_transcript: notes,
          analysis: summary,
          mindmap: mindmapData,
          title: title,
        }),
      });
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`保存に失敗しました: ${errorText}`);
      }
      const result = await res.json();
      if (result.status === 'success') {
        setSaveStatus('保存が完了しました！');
        // 保存成功後は自動的にモーダルを閉じ、保存済み一覧を更新
        setShowSaveModal(false);
        fetchSavedMinutes();
      } else {
        setSaveStatus('保存に失敗しました: ' + (result.detail || ''));
      }
    } catch (err) {
      console.error(err);
      setSaveStatus(err.message || '保存中にエラーが発生しました');
    }
  };

  // --------------------------
  // モーダル閉じる
  // --------------------------
  const closeSaveModal = () => {
    setSaveStatus(null);
  };

  // --------------------------
  // サイドバーの開閉
  // --------------------------
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <div style={styles.appContainer}>
      {/* キーアニメーション定義 */}
      <style>{`
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slideDown { from { transform: translateY(-20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        @keyframes popIn { from { transform: scale(0.9); opacity: 0; } to { transform: scale(1); opacity: 1; } }
      `}</style>

      {/* ヘッダー */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <button style={styles.toggleButton} onClick={toggleSidebar}>
            {sidebarOpen ? '←' : '→'}
          </button>
          <h1 style={styles.logo}>Conect AI</h1>
        </div>
        <div style={styles.headerRight}>
          {/* 必要に応じてナビゲーションボタンなど */}
        </div>
      </header>

      {/* メインレイアウト */}
      <div style={styles.contentWrapper}>
        {/* サイドバー */}
        {sidebarOpen && (
          <aside style={styles.sidebar}>
            <h2 style={styles.sidebarTitle}>保存した議事録</h2>
            {savedMinutes.length > 0 ? (
              savedMinutes.map((minute) => (
                <div
                  key={minute.id}
                  style={styles.sidebarItem}
                  onClick={() => {
                    setSelectedMinute(minute);
                    setShowSaveModal(false);
                  }}
                >
                  {minute.title}
                </div>
              ))
            ) : (
              <p style={styles.sidebarText}>保存された議事録はありません</p>
            )}
          </aside>
        )}

        {/* メインコンテンツ */}
        <main style={{ ...styles.mainContent, marginLeft: sidebarOpen ? 300 : 20 }}>
          {selectedMinute ? (
            // 保存済み議事録詳細表示
            <div style={styles.detailContainer}>
              <button style={styles.backButton} onClick={() => setSelectedMinute(null)}>
                戻る
              </button>
              <h2 style={styles.detailTitle}>{selectedMinute.title}</h2>
              <div style={styles.card}>
                <h3 style={styles.cardTitle}>整形された文字起こし</h3>
                <p style={styles.cardContent}>{selectedMinute.formatted_transcript}</p>
              </div>
              <div style={styles.card}>
                <h3 style={styles.cardTitle}>議事録・改善案</h3>
                <p style={styles.cardContent}>{selectedMinute.analysis}</p>
              </div>
              {selectedMinute.mindmap && (
                <div style={styles.mindmapContainer}>
                  <Tree
                    data={selectedMinute.mindmap}
                    orientation="vertical"
                    translate={{ x: 400, y: 50 }}
                    pathFunc="diagonal"
                    separation={{ siblings: 1.5, nonSiblings: 2 }}
                    nodeSize={{ x: 300, y: 100 }}
                  />
                </div>
              )}
            </div>
          ) : (
            // 新規録音＆生成結果表示＋編集可能な入力欄
            <div style={styles.recorderContainer}>
              <div style={styles.descriptionBox}>
                <p style={styles.descriptionText}>
                  このアプリでは、会議の録音、文字起こし、議事録・改善案生成、そしてマインドマップ作成が可能です。<br />
                  「録音開始」ボタンを押して会議内容を録音し、録音停止後に自動生成された議事録が下に表示されます。<br />
                  ※表示された内容は、必要に応じて編集できます。
                </p>
              </div>
              <ReactMediaRecorder
                audio
                video={false}
                mimeType="audio/webm;codecs=opus"
                onStop={(blobUrl, blob) => {
                  sendAudioToServer(blob);
                }}
                render={({ status, startRecording, stopRecording }) => (
                  <div style={styles.controlsWrapper}>
                    {!recording ? (
                      <button
                        style={styles.actionButton}
                        onClick={() => {
                          startRecording();
                          setRecording(true);
                          setRecorded(false);
                        }}
                      >
                        録音開始
                      </button>
                    ) : (
                      <button
                        style={styles.actionButton}
                        onClick={() => {
                          stopRecording();
                          setRecording(false);
                          setRecorded(true);
                        }}
                      >
                        録音停止
                      </button>
                    )}
                  </div>
                )}
              />
              {recorded && (
                <div style={styles.generatedContainer}>
                  <div style={styles.generatedContent}>
                    <h2>生成された議事録（編集可能）</h2>
                    <div style={{ marginBottom: '10px' }}>
                      <label>タイトル: </label>
                      <input
                        type="text"
                        value={title}
                        onChange={(e) => setTitle(e.target.value)}
                        style={styles.editableTitle}
                      />
                    </div>
                    <div style={{ marginBottom: '10px' }}>
                      <label>整形された文字起こし: </label>
                      <textarea
                        value={notes}
                        onChange={(e) => setNotes(e.target.value)}
                        style={styles.editableTextarea}
                      />
                    </div>
                    <div style={{ marginBottom: '10px' }}>
                      <label>議事録・改善案: </label>
                      <textarea
                        value={summary}
                        onChange={(e) => setSummary(e.target.value)}
                        style={styles.editableTextarea}
                      />
                    </div>
                    {mindmapData && (
                      <div style={styles.mindmapContainer}>
                        <Tree
                          data={mindmapData}
                          orientation="vertical"
                          translate={{ x: 400, y: 50 }}
                          pathFunc="diagonal"
                          separation={{ siblings: 1.5, nonSiblings: 2 }}
                          nodeSize={{ x: 300, y: 100 }}
                        />
                      </div>
                    )}
                  </div>
                  {/* 保存するボタン（編集後の内容で保存するためのモーダルを表示） */}
                  <button
                    style={styles.actionButton}
                    onClick={() => setShowSaveModal(true)}
                  >
                    確認画面へ
                  </button>
                </div>
              )}
              {/* 議事録保存用モーダル（確認用） */}
              {showSaveModal && (
                <div style={styles.modalOverlay}>
                  <div style={styles.modalContent}>
                    <h2 style={styles.modalTitle}>この内容で保存します</h2>
                    <div style={styles.modalSection}>
                      <h3 style={styles.modalSubTitle}>タイトル</h3>
                      <p style={styles.modalText}>{title}</p>
                    </div>
                    <div style={styles.modalSection}>
                      <h3 style={styles.modalSubTitle}>整形された文字起こし</h3>
                      <p style={styles.modalText}>{notes}</p>
                    </div>
                    <div style={styles.modalSection}>
                      <h3 style={styles.modalSubTitle}>議事録・改善案</h3>
                      <p style={styles.modalText}>{summary}</p>
                    </div>
                    {mindmapData && (
                      <div style={styles.modalSection}>
                        <h3 style={styles.modalSubTitle}>マインドマップ</h3>
                        <div style={styles.mindmapModalContainer}>
                          <Tree
                            data={mindmapData}
                            orientation="vertical"
                            translate={{ x: 350, y: 50 }}
                            pathFunc="diagonal"
                            separation={{ siblings: 1.5, nonSiblings: 2 }}
                            nodeSize={{ x: 300, y: 100 }}
                          />
                        </div>
                      </div>
                    )}
                    <button
                      style={styles.modalActionButton}
                      onClick={saveToDatabase}
                    >
                      この内容で保存する
                    </button>
                    <button
                      style={styles.modalCloseButton}
                      onClick={() => setShowSaveModal(false)}
                    >
                      閉じる
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </main>
      </div>
      {/* ローディングモーダル */}
      {loading && (
        <div style={styles.modalOverlay}>
          <div style={styles.modalContent}>
            <p style={styles.modalText}>議事録を生成中です...お待ちください</p>
          </div>
        </div>
      )}
      {/* 保存結果モーダル */}
      {saveStatus && (
        <div style={styles.modalOverlay}>
          <div style={styles.modalContent}>
            <p style={styles.modalText}>{saveStatus}</p>
            <button style={styles.modalCloseButton} onClick={closeSaveModal}>
              閉じる
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// --------------------------
// スタイル定義
// --------------------------
const styles = {
  appContainer: {
    backgroundColor: '#fff',
    minHeight: '100vh',
    fontFamily: '"Montserrat", "Playfair Display", sans-serif',
    color: '#333',
    padding: '20px',
    transition: 'all 0.3s ease',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: '10px 20px',
    borderBottom: '1px solid #ddd',
    position: 'sticky',
    top: 0,
    zIndex: 10,
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '15px',
  },
  toggleButton: {
    backgroundColor: '#4facfe',
    border: 'none',
    borderRadius: '50%',
    width: '36px',
    height: '36px',
    color: '#fff',
    fontSize: '1.2rem',
    cursor: 'pointer',
    transition: 'background 0.3s ease',
  },
  logo: {
    fontSize: '1.8rem',
    margin: 0,
    fontWeight: 'bold',
  },
  headerRight: {
    display: 'flex',
    gap: '10px',
  },
  contentWrapper: {
    display: 'flex',
    position: 'relative',
    transition: 'all 0.3s ease',
  },
  sidebar: {
    width: '280px',
    backgroundColor: '#f7f7f7',
    padding: '20px',
    height: 'calc(100vh - 60px)',
    overflowY: 'auto',
    borderRight: '1px solid #eee',
    position: 'fixed',
    left: 0,
    top: 60,
    bottom: 0,
    transition: 'transform 0.3s ease',
  },
  sidebarTitle: {
    fontSize: '1.4rem',
    marginBottom: '15px',
    fontWeight: 'bold',
  },
  sidebarItem: {
    padding: '10px',
    borderBottom: '1px solid #ddd',
    cursor: 'pointer',
    transition: 'background 0.3s ease',
  },
  sidebarText: {
    fontSize: '0.9rem',
    color: '#666',
  },
  mainContent: {
    marginLeft: 300,
    padding: '20px',
    width: '100%',
    transition: 'margin-left 0.3s ease',
  },
  recorderContainer: {
    padding: '20px',
  },
  descriptionBox: {
    marginBottom: '30px',
    animation: 'fadeIn 0.5s ease',
  },
  descriptionText: {
    fontSize: '1.1rem',
    lineHeight: '1.6',
  },
  controlsWrapper: {
    display: 'flex',
    justifyContent: 'center',
    gap: '20px',
    animation: 'fadeIn 0.5s ease',
  },
  actionButton: {
    background: 'linear-gradient(45deg, #4facfe, #00f2fe)',
    border: 'none',
    padding: '15px 30px',
    borderRadius: '30px',
    cursor: 'pointer',
    fontSize: '1.3rem',
    color: '#fff',
    transition: 'transform 0.2s ease, background 0.3s ease',
  },
  generatedContainer: {
    marginTop: '30px',
    animation: 'fadeIn 0.5s ease',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '20px',
  },
  generatedContent: {
    width: '100%',
  },
  // 編集可能な入力欄用スタイル
  editableTitle: {
    width: '100%',
    padding: '10px 15px',
    fontSize: '1.2rem',
    borderRadius: '5px',
    border: '1px solid #ccc',
    marginTop: '5px',
  },
  editableTextarea: {
    width: '100%',
    padding: '10px 15px',
    fontSize: '1rem',
    borderRadius: '5px',
    border: '1px solid #ccc',
    minHeight: '100px',
    marginTop: '5px',
    resize: 'vertical',
  },
  modalOverlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.4)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
    animation: 'fadeIn 0.3s ease',
  },
  modalContent: {
    backgroundColor: '#fff',
    padding: '30px',
    borderRadius: '10px',
    width: '80%',
    height: '80%',
    maxWidth: '900px',
    textAlign: 'left',
    boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
    animation: 'slideDown 0.4s ease, popIn 0.4s ease',
    display: 'flex',
    flexDirection: 'column',
    overflowY: 'auto',
    boxSizing: 'border-box',
  },
  modalTitle: {
    fontSize: '1.8rem',
    marginBottom: '20px',
    textAlign: 'center',
    fontWeight: 'bold',
    flexShrink: 0,
  },
  titleInput: {
    width: '100%',
    padding: '12px 20px',
    marginBottom: '20px',
    borderRadius: '30px',
    border: '1px solid #ccc',
    fontSize: '1rem',
    outline: 'none',
    transition: 'border-color 0.3s ease',
    boxSizing: 'border-box',
  },
  modalSection: {
    marginBottom: '20px',
    flexShrink: 0,
  },
  modalSubTitle: {
    fontSize: '1.3rem',
    marginBottom: '10px',
    fontWeight: 'bold',
  },
  modalText: {
    fontSize: '1rem',
    lineHeight: 1.6,
  },
  mindmapModalContainer: {
    width: '100%',
    minHeight: '300px',
    border: '1px solid #eee',
    borderRadius: '8px',
    position: 'relative',
    overflow: 'auto',
    height: '300px',
    backgroundColor: '#fafafa',
  },
  modalActionButton: {
    background: 'linear-gradient(45deg, #4facfe, #00f2fe)',
    border: 'none',
    padding: '12px 25px',
    borderRadius: '30px',
    cursor: 'pointer',
    fontSize: '1.1rem',
    color: '#fff',
    width: '100%',
    marginBottom: '10px',
    transition: 'background 0.3s ease, transform 0.2s ease',
    flexShrink: 0,
  },
  modalCloseButton: {
    backgroundColor: '#ccc',
    border: 'none',
    padding: '12px 25px',
    borderRadius: '30px',
    cursor: 'pointer',
    fontSize: '1rem',
    color: '#333',
    width: '100%',
    transition: 'background 0.3s ease, transform 0.2s ease',
    flexShrink: 0,
  },
  detailContainer: {
    animation: 'fadeIn 0.3s ease',
  },
  detailTitle: {
    fontSize: '1.8rem',
    fontWeight: 'bold',
    marginBottom: '20px',
  },
  card: {
    backgroundColor: '#fff',
    padding: '20px',
    borderRadius: '8px',
    margin: '10px 0',
    boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
  },
  cardTitle: {
    fontSize: '1.3rem',
    marginBottom: '10px',
    fontWeight: 'bold',
  },
  cardContent: {
    fontSize: '1rem',
    lineHeight: '1.6',
  },
  mindmapContainer: {
    height: '400px',
    overflow: 'auto',
    marginTop: '20px',
    border: '1px solid #eee',
    borderRadius: '8px',
    backgroundColor: '#fafafa',
  },
  backButton: {
    backgroundColor: '#4facfe',
    border: 'none',
    padding: '10px 20px',
    borderRadius: '30px',
    cursor: 'pointer',
    fontSize: '1rem',
    color: '#fff',
    marginBottom: '20px',
    transition: 'background 0.3s ease',
  },
};

export default App;
