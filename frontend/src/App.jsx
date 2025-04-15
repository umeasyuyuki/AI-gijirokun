import React, { useState, useEffect, useRef } from 'react';
import Tree from 'react-d3-tree';

// ======== MUI & その他 ========
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Drawer,
  List,
  ListItemButton,
  ListItemText,
  Box,
  Paper,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  CircularProgress,
  IconButton
} from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { motion, AnimatePresence } from 'framer-motion';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';

/** テーマ設定 */
const theme = createTheme({
  palette: {
    primary: { main: '#1976d2' },
    secondary: { main: '#00f2fe' },
  },
  typography: {
    fontFamily: [
      'Montserrat',
      'Playfair Display',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
  },
});

// アニメーション用バリアント
const fadeInVariants = {
  hidden: { opacity: 0, y: -10 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
};

/**
 * 枝のカスタム描画用コンポーネント
 * ここで linkDatum.source.depth などを使って色を分岐させる
 */
const colorPalette = ['#cc00cc', '#ff4500', '#00aaff', '#ff00ff', '#ff8c00', '#228b22'];
function CustomColoredLink({ linkDatum, orientation }) {
  // 階層（depth）に応じて色を切り替えたり、あるいは子ノードの情報で分岐させてもOK
  const strokeColor = colorPalette[linkDatum.source.depth % colorPalette.length];

  // "diagonal" パス計算（横向きレイアウト用）
  // ここでは簡易的にベジェ曲線で左右に伸びる対角線を作成
  const { source, target } = linkDatum;
  // x=横軸, y=縦軸で、横向きの場合は X が左右, Y が上下
  const path = `M${source.x},${source.y}
                C${(source.x + target.x) / 2},${source.y},
                ${(source.x + target.x) / 2},${target.y},
                ${target.x},${target.y}`;

  return (
    <path
      d={path}
      fill="none"
      stroke={strokeColor}
      strokeWidth="3"
    />
  );
}

function App() {
  const API_URL = "http://localhost:8000";

  // --------------------------
  // ステート類（議事録表示・生成）
  // --------------------------
  const [notes, setNotes] = useState('ここに文字起こしが表示されます');
  const [summary, setSummary] = useState('ここに議事録・改善案が表示されます');
  const [mindmapData, setMindmapData] = useState(null);
  const [title, setTitle] = useState('');

  const [recording, setRecording] = useState(false);
  const [recorded, setRecorded] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [loading, setLoading] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null);

  // 保存済み議事録
  const [savedMinutes, setSavedMinutes] = useState([]);
  const [selectedMinute, setSelectedMinute] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // 録音関連
  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const [audioChunks, setAudioChunks] = useState([]);
  const localChunksRef = useRef([]);

  // 録音時間カウンター
  const [recordingTime, setRecordingTime] = useState(0);
  const recordingTimerRef = useRef(null);

  // モード: audio / text / chat
  const [mode, setMode] = useState("audio");
  const [inputTranscript, setInputTranscript] = useState("");

  // チャットモード
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  // --------------------------
  // 1) 保存済み議事録を取得
  // --------------------------
  const fetchSavedMinutes = async () => {
    try {
      const res = await fetch(`${API_URL}/get-minutes`);
      if (!res.ok) throw new Error('保存された議事録の取得に失敗しました');
      const data = await res.json();
      const sorted = data.minutes.sort((a, b) => b.id - a.id);
      setSavedMinutes(sorted);
    } catch (error) {
      console.error('Error fetching saved minutes:', error);
    }
  };

  useEffect(() => {
    fetchSavedMinutes();
  }, []);

  /**
   * ブラウザがサポートする録音mimeTypeを順に探す
   */
  function getSupportedMimeType() {
    const possibleTypes = [
      "audio/webm;codecs=opus",
      "audio/ogg;codecs=opus",
      "audio/webm",
      "audio/ogg"
    ];
    for (const t of possibleTypes) {
      if (MediaRecorder.isTypeSupported(t)) {
        return t;
      }
    }
    return "";
  }

  // 録音時間を mm:ss フォーマットに変換
  const formatRecordingTime = (seconds) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  // ====================================================
  // 録音開始
  // ====================================================
  const startRecording = async () => {
    setRecording(true);
    setRecorded(false);
    setAudioChunks([]);
    localChunksRef.current = [];
    setNotes('録音が開始されました');
    setSummary('会議中...');
    setRecordingTime(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      // 録音時間カウント開始
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

      const mimeType = getSupportedMimeType() || "audio/webm";
      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          localChunksRef.current.push(e.data);
        }
      };
      mediaRecorder.onstop = async () => {
        // タイマー停止
        if (recordingTimerRef.current) {
          clearInterval(recordingTimerRef.current);
          recordingTimerRef.current = null;
        }

        const blob = new Blob(localChunksRef.current, { type: mimeType });
        localChunksRef.current = [];
        setAudioChunks((prev) => [...prev, blob]);
        console.log("🎙️ 録音終了: Blob 作成済み", blob.size);

        setRecording(false);
        setRecorded(true);
        setNotes("録音が停止しました。文字起こしを準備中...");

        // サーバー送信
        await sendAudioToServer(blob);
      };
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      console.log("🎙️ 録音開始");
    } catch (err) {
      console.error('録音開始失敗:', err);
      setNotes('録音開始に失敗しました: ' + err.message);
      setRecording(false);
    }
  };

  // ====================================================
  // 録音停止
  // ====================================================
  const stopRecording = async () => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
    }
    console.log("🛑 録音停止要求");
  };

  // --------------------------
  // 録音データをサーバーに送信
  // --------------------------
  const sendAudioToServer = async (audioBlob) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('audio', new File([audioBlob], 'recording.webm', { type: audioBlob.type }));

      console.log("📤 /transcribe へ POST. blobSize=", audioBlob.size);
      const res = await fetch(`${API_URL}/transcribe`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        throw new Error(`サーバーエラー: ${res.statusText}`);
      }
      const data = await res.json();
      console.log("🧠 文字起こし & 議事録結果:", data);

      setNotes(data.formatted_transcript);
      setSummary(data.analysis);
      setMindmapData(data.mindmap);
      setTitle(data.title);
    } catch (err) {
      console.error('通信エラー:', err);
      setNotes('通信エラーが発生しました');
    } finally {
      setLoading(false);
    }
  };

  // ====================================================
  // テキストモード: 議事録生成
  // ====================================================
  const sendTextToServer = async () => {
    if (!inputTranscript.trim()) {
      alert("テキストが空です。");
      return;
    }
    try {
      setLoading(true);
      const res = await fetch(`${API_URL}/transcribe-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ raw_text: inputTranscript }),
      });
      if (!res.ok) throw new Error(`サーバーエラー: ${res.statusText}`);
      const data = await res.json();

      setNotes(data.formatted_transcript);
      setSummary(data.analysis);
      setMindmapData(data.mindmap);
      setTitle(data.title);
      setRecorded(true);
    } catch (err) {
      console.error('テキストモード失敗:', err);
      setNotes('テキスト議事録の生成中にエラーが発生しました。');
    } finally {
      setLoading(false);
    }
  };

  // --------------------------
  // 保存・削除など
  // --------------------------
  const saveToDatabase = async () => {
    try {
      const res = await fetch(`${API_URL}/save-minutes`, {
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
        setShowSaveModal(false);
        fetchSavedMinutes();
      } else {
        setSaveStatus('保存に失敗しました: ' + (result.detail || ''));
      }
    } catch (err) {
      console.error('保存失敗:', err);
      setSaveStatus(err.message || '保存中にエラーが発生しました');
    }
  };

  const deleteMinutes = async (id) => {
    if (!window.confirm("本当に削除しますか？")) return;
    try {
      const res = await fetch(`${API_URL}/delete-minutes/${id}`, {
        method: 'DELETE',
      });
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`削除に失敗しました: ${errorText}`);
      }
      const result = await res.json();
      if (result.status === 'success') {
        alert('削除しました');
        fetchSavedMinutes();
        if (selectedMinute && selectedMinute.id === id) {
          setSelectedMinute(null);
        }
      } else {
        alert('削除に失敗: ' + result.detail);
      }
    } catch (err) {
      console.error('削除エラー:', err);
      alert('削除中にエラーが発生しました');
    }
  };

  const handleCopyContent = async (content) => {
    try {
      await navigator.clipboard.writeText(content);
      alert('コピーしました!');
    } catch (err) {
      console.error('コピー失敗:', err);
      alert('コピーに失敗しました');
    }
  };

  // --------------------------
  // チャットモード
  // --------------------------
  const sendChatMessage = async () => {
    if (!chatInput.trim()) return;
    const userMsg = { sender: 'user', text: chatInput };
    setChatMessages((prev) => [...prev, userMsg]);
    setChatInput("");
    try {
      setChatLoading(true);
      const res = await fetch(`${API_URL}/chatbot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg.text }),
      });
      if (!res.ok) {
        throw new Error(`サーバーエラー: ${res.statusText}`);
      }
      const data = await res.json();
      const botMsg = { sender: 'bot', text: data.response };
      setChatMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      console.error("チャットbotエラー:", err);
      const botMsg = { sender: 'bot', text: "エラーが発生しました。" };
      setChatMessages((prev) => [...prev, botMsg]);
    } finally {
      setChatLoading(false);
    }
  };

  // その他
  const closeSaveModal = () => {
    setSaveStatus(null);
  };
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <ThemeProvider theme={theme}>
      <Box
        sx={{
          minHeight: '100vh',
          background: 'linear-gradient(to bottom right, #f0f9ff, #cfe9ff, #eef6ff)',
          position: 'relative',
        }}
      >
        <Button
          variant="contained"
          color="primary"
          onClick={toggleSidebar}
          sx={{
            position: 'absolute',
            left: 10,
            top: 20,
            borderRadius: '50%',
            width: 45,
            minWidth: 45,
            height: 45,
            fontSize: '1rem',
            zIndex: 2000,
          }}
        >
          {sidebarOpen ? '←' : '→'}
        </Button>
        <AppBar position="sticky" sx={{ bgcolor: 'rgba(255,255,255,0.8)', backdropFilter: 'blur(8px)' }}>
          <Toolbar sx={{ justifyContent: 'center' }}>
            <Typography variant="h4" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
              Conect AI
            </Typography>
          </Toolbar>
        </AppBar>
        <Drawer
          variant="persistent"
          anchor="left"
          open={sidebarOpen}
          sx={{
            '& .MuiDrawer-paper': {
              width: 240,
              boxSizing: 'border-box',
              borderRight: '1px solid #eee',
              bgcolor: 'rgba(255,255,255,0.85)',
              backdropFilter: 'blur(6px)',
              pt: 8,
            },
          }}
        >
          <Box sx={{ p: 2 }}>
            <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 'bold', color: '#1976d2' }}>
              保存した議事録
            </Typography>
            <List>
              {savedMinutes.length > 0 ? (
                savedMinutes.map((minute) => (
                  <ListItemButton
                    key={minute.id}
                    onClick={() => {
                      setSelectedMinute(minute);
                      setShowSaveModal(false);
                    }}
                  >
                    <ListItemText primary={minute.title} />
                  </ListItemButton>
                ))
              ) : (
                <ListItemButton>
                  <ListItemText primary="保存された議事録はありません" />
                </ListItemButton>
              )}
            </List>
          </Box>
        </Drawer>
        <Box
          sx={{
            ml: sidebarOpen ? '240px' : 2,
            p: 3,
            pt: 10,
            transition: 'margin-left 0.3s ease',
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mb: 3 }}>
            <Button
              variant={mode === 'audio' ? 'contained' : 'outlined'}
              sx={{ borderRadius: 3, px: 4, py: 1, fontWeight: 'bold' }}
              onClick={() => setMode('audio')}
            >
              音声モード
            </Button>
            <Button
              variant={mode === 'text' ? 'contained' : 'outlined'}
              sx={{ borderRadius: 3, px: 4, py: 1, fontWeight: 'bold' }}
              onClick={() => setMode('text')}
            >
              テキストモード
            </Button>
            <Button
              variant={mode === 'chat' ? 'contained' : 'outlined'}
              sx={{ borderRadius: 3, px: 4, py: 1, fontWeight: 'bold' }}
              onClick={() => setMode('chat')}
            >
              チャットモード
            </Button>
          </Box>
          <AnimatePresence>
            {selectedMinute ? (
              // --------------------------
              // 保存済み議事録の詳細表示
              // --------------------------
              <motion.div
                key="detailView"
                variants={fadeInVariants}
                initial="hidden"
                animate="visible"
                exit={{ opacity: 0 }}
                style={{ position: 'relative' }}
              >
                <Button
                  variant="contained"
                  onClick={() => setSelectedMinute(null)}
                  sx={{
                    position: 'absolute',
                    left: 10,
                    top: 10,
                    borderRadius: 3,
                    px: 4,
                    py: 1,
                    fontWeight: 'bold',
                    zIndex: 1000,
                  }}
                >
                  戻る
                </Button>
                <Typography
                  variant="h5"
                  sx={{
                    fontWeight: 'bold',
                    color: 'primary.main',
                    textAlign: 'center',
                    mb: 4,
                  }}
                >
                  {selectedMinute.title}
                </Typography>
                <Paper sx={{ p: 3, mb: 3, position: 'relative', borderRadius: 3 }}>
                  <Typography
                    variant="subtitle1"
                    sx={{
                      fontWeight: 'bold',
                      color: 'secondary.main',
                      mb: 2,
                      fontSize: '1.2rem',
                    }}
                  >
                    整形された文字起こし
                  </Typography>
                  <IconButton
                    size="small"
                    sx={{ position: 'absolute', top: 8, right: 8 }}
                    onClick={() => handleCopyContent(selectedMinute.formatted_transcript)}
                  >
                    <ContentCopyIcon fontSize="small" />
                  </IconButton>
                  <Typography
                    variant="body1"
                    sx={{ fontSize: '1.1rem', fontWeight: 'bold', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}
                  >
                    {selectedMinute.formatted_transcript}
                  </Typography>
                </Paper>
                <Paper sx={{ p: 3, mb: 3, position: 'relative', borderRadius: 3 }}>
                  <Typography
                    variant="subtitle1"
                    sx={{
                      fontWeight: 'bold',
                      color: 'secondary.main',
                      mb: 2,
                      fontSize: '1.2rem',
                    }}
                  >
                    議事録・改善案
                  </Typography>
                  <IconButton
                    size="small"
                    sx={{ position: 'absolute', top: 8, right: 8 }}
                    onClick={() => handleCopyContent(selectedMinute.analysis)}
                  >
                    <ContentCopyIcon fontSize="small" />
                  </IconButton>
                  <Typography
                    variant="body1"
                    sx={{ fontSize: '1.1rem', fontWeight: 'bold', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}
                  >
                    {selectedMinute.analysis}
                  </Typography>
                </Paper>
                {selectedMinute.mindmap && (
                  <Paper sx={{ p: 3, mb: 3, borderRadius: 3 }}>
                    <Typography
                      variant="subtitle1"
                      sx={{
                        fontWeight: 'bold',
                        color: 'secondary.main',
                        mb: 2,
                        fontSize: '1.2rem',
                      }}
                    >
                      マインドマップ
                    </Typography>
                    <Box sx={{ border: '1px solid #ccc', borderRadius: 2, height: '600px', overflow: 'auto' }}>
                      {/*
                        orientation="horizontal" で左右に広がる
                        renderCustomLinkElement でカラフルな枝を描画
                        translate で中央寄せ
                      */}
                      <Tree
                        data={selectedMinute.mindmap}
                        orientation="horizontal"
                        translate={{ x: 400, y: 300 }}
                        renderCustomLinkElement={(rd3tProps) => (
                          <CustomColoredLink {...rd3tProps} />
                        )}
                        // separation でノード間隔を調整
                        separation={{ siblings: 1.3, nonSiblings: 1.4 }}
                        // デフォルトパスの代わりに自分で描画するので pathFunc は無効でもOK
                        pathFunc="diagonal"
                        // ノードサイズを大きめに
                        nodeSize={{ x: 180, y: 200 }}
                      />
                    </Box>
                  </Paper>
                )}
                <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                  <Button
                    variant="contained"
                    color="error"
                    onClick={() => deleteMinutes(selectedMinute.id)}
                    sx={{ borderRadius: 3, px: 4, py: 1, fontWeight: 'bold' }}
                  >
                    この議事録を削除
                  </Button>
                </Box>
              </motion.div>
            ) : (
              // --------------------------
              // 新規録音 or テキスト or チャット
              // --------------------------
              <motion.div key={mode} variants={fadeInVariants} initial="hidden" animate="visible" exit={{ opacity: 0 }}>
                {mode === "audio" && (
                  <Paper sx={{ p: 4, mb: 4, borderRadius: 3 }} elevation={3}>
                    <Typography variant="h5" sx={{ textAlign: 'center', mb: 3, color: 'primary.main', fontWeight: 'bold' }}>
                      音声モード
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                      {!recording ? (
                        <Button
                          variant="contained"
                          color="primary"
                          onClick={startRecording}
                          sx={{ borderRadius: 3, px: 4, py: 1, fontWeight: 'bold' }}
                        >
                          録音開始
                        </Button>
                      ) : (
                        <Button
                          variant="contained"
                          color="secondary"
                          onClick={stopRecording}
                          sx={{ borderRadius: 3, px: 4, py: 1, fontWeight: 'bold' }}
                        >
                          録音停止
                        </Button>
                      )}
                      {recording && (
                        <Typography variant="body1" sx={{ mt: 1, fontWeight: 'bold', fontSize: '1rem' }}>
                          録音中... ({formatRecordingTime(recordingTime)})
                        </Typography>
                      )}
                    </Box>

                    {/* 録音後のプレビュー */}
                    {recorded && (
                      <Box sx={{ mt: 4 }}>
                        <Typography variant="h6" sx={{ mb: 2, color: 'primary.main', fontWeight: 'bold' }}>
                          生成結果プレビュー
                        </Typography>
                        <TextField
                          label="タイトル"
                          fullWidth
                          variant="outlined"
                          size="small"
                          value={title}
                          onChange={(e) => setTitle(e.target.value)}
                          sx={{ mb: 2 }}
                        />
                        <Typography sx={{ fontWeight: 'bold', mb: 1 }}>文字起こし</Typography>
                        <TextField
                          multiline
                          rows={3}
                          fullWidth
                          variant="outlined"
                          size="small"
                          value={notes}
                          onChange={(e) => setNotes(e.target.value)}
                          sx={{ mb: 2 }}
                        />

                        <Typography sx={{ fontWeight: 'bold', mb: 1 }}>議事録・改善案</Typography>
                        <TextField
                          multiline
                          rows={3}
                          fullWidth
                          variant="outlined"
                          size="small"
                          value={summary}
                          onChange={(e) => setSummary(e.target.value)}
                          sx={{ mb: 2 }}
                        />

                        {mindmapData && (
                          <Box sx={{ border: '1px solid #ccc', borderRadius: 2, height: 300, overflow: 'auto', p: 1 }}>
                            {/* 
                              同じくプレビュー側でも左右に伸びる形 + カラフルリンク 
                            */}
                            <Tree
                              data={mindmapData}
                              orientation="horizontal"
                              translate={{ x: 300, y: 150 }}
                              renderCustomLinkElement={(rd3tProps) => (
                                <CustomColoredLink {...rd3tProps} />
                              )}
                              separation={{ siblings: 1.3, nonSiblings: 1.4 }}
                              pathFunc="diagonal"
                              nodeSize={{ x: 180, y: 150 }}
                            />
                          </Box>
                        )}

                        <Button
                          variant="contained"
                          color="success"
                          sx={{ mt: 2, borderRadius: 3, px: 4, py: 1, fontWeight: 'bold' }}
                          onClick={() => setShowSaveModal(true)}
                        >
                          確認画面へ
                        </Button>
                      </Box>
                    )}
                  </Paper>
                )}

                {mode === "text" && (
                  <Paper sx={{ p: 4, mb: 4, borderRadius: 3 }} elevation={3}>
                    <Typography variant="h5" sx={{ textAlign: 'center', mb: 3, color: 'primary.main', fontWeight: 'bold' }}>
                      テキストモード
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                      <TextField
                        fullWidth
                        multiline
                        rows={4}
                        variant="outlined"
                        size="small"
                        sx={{ maxWidth: '600px', fontSize: '1.1rem', fontWeight: 'bold', borderRadius: 2 }}
                        value={inputTranscript}
                        onChange={(e) => setInputTranscript(e.target.value)}
                        placeholder="ここにテキストを貼り付けてください"
                      />
                      <Button
                        variant="contained"
                        color="primary"
                        onClick={sendTextToServer}
                        sx={{ borderRadius: 3, px: 4, py: 1, fontWeight: 'bold' }}
                      >
                        議事録生成
                      </Button>
                    </Box>

                    {recorded && (
                      <Box sx={{ mt: 4 }}>
                        <Typography variant="h6" sx={{ mb: 2, color: 'primary.main', fontWeight: 'bold' }}>
                          生成結果プレビュー
                        </Typography>
                        <TextField
                          label="タイトル"
                          fullWidth
                          variant="outlined"
                          size="small"
                          value={title}
                          onChange={(e) => setTitle(e.target.value)}
                          sx={{ mb: 2 }}
                        />
                        <Typography sx={{ fontWeight: 'bold', mb: 1 }}>文字起こし</Typography>
                        <TextField
                          multiline
                          rows={3}
                          fullWidth
                          variant="outlined"
                          size="small"
                          value={notes}
                          onChange={(e) => setNotes(e.target.value)}
                          sx={{ mb: 2 }}
                        />

                        <Typography sx={{ fontWeight: 'bold', mb: 1 }}>議事録・改善案</Typography>
                        <TextField
                          multiline
                          rows={3}
                          fullWidth
                          variant="outlined"
                          size="small"
                          value={summary}
                          onChange={(e) => setSummary(e.target.value)}
                          sx={{ mb: 2 }}
                        />

                        {mindmapData && (
                          <Box sx={{ border: '1px solid #ccc', borderRadius: 2, height: 300, overflow: 'auto', p: 1 }}>
                            <Tree
                              data={mindmapData}
                              orientation="horizontal"
                              translate={{ x: 250, y: 150 }}
                              renderCustomLinkElement={(rd3tProps) => (
                                <CustomColoredLink {...rd3tProps} />
                              )}
                              separation={{ siblings: 1.3, nonSiblings: 1.4 }}
                              pathFunc="diagonal"
                              nodeSize={{ x: 180, y: 150 }}
                            />
                          </Box>
                        )}

                        <Button
                          variant="contained"
                          color="success"
                          sx={{ mt: 2, borderRadius: 3, px: 4, py: 1, fontWeight: 'bold' }}
                          onClick={() => setShowSaveModal(true)}
                        >
                          確認画面へ
                        </Button>
                      </Box>
                    )}
                  </Paper>
                )}

                {mode === "chat" && (
                  <Paper sx={{ p: 4, mb: 4, borderRadius: 3 }} elevation={3}>
                    <Typography variant="h5" sx={{ textAlign: 'center', mb: 3, color: 'primary.main', fontWeight: 'bold' }}>
                      チャットモード
                    </Typography>
                    <Box
                      sx={{
                        display: 'flex',
                        flexDirection: 'column',
                        height: '500px',
                        maxWidth: '700px',
                        margin: '0 auto',
                        border: '1px solid #ccc',
                        borderRadius: 3,
                        overflow: 'hidden',
                      }}
                    >
                      <Box
                        sx={{
                          flex: 1,
                          overflowY: 'auto',
                          p: 2,
                          display: 'flex',
                          flexDirection: 'column',
                          gap: 1,
                          backgroundColor: '#fdfdfd',
                        }}
                      >
                        {chatMessages.map((msg, idx) => (
                          <Box key={idx} sx={{ alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start', maxWidth: '80%' }}>
                            <Paper
                              sx={{
                                p: 1.5,
                                my: 0.5,
                                borderRadius: 2,
                                backgroundColor: msg.sender === 'user' ? '#e3f2fd' : '#ffffff',
                                boxShadow: 2,
                              }}
                            >
                              <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', fontSize: '1rem' }}>
                                {msg.text}
                              </Typography>
                            </Paper>
                          </Box>
                        ))}
                        {chatLoading && (
                          <Box sx={{ alignSelf: 'flex-start', my: 1 }}>
                            <Paper sx={{ p: 1.5, borderRadius: 2, backgroundColor: '#fff', boxShadow: 2 }}>
                              <Typography variant="body1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <CircularProgress size={20} />
                                生成中...
                              </Typography>
                            </Paper>
                          </Box>
                        )}
                      </Box>
                      <Box
                        sx={{
                          borderTop: '1px solid #ccc',
                          p: 2,
                          display: 'flex',
                          gap: 1,
                          backgroundColor: '#f0f0f0',
                        }}
                      >
                        <TextField
                          variant="outlined"
                          size="small"
                          fullWidth
                          value={chatInput}
                          onChange={(e) => setChatInput(e.target.value)}
                          placeholder="質問を入力...(Enterで改行。送信はボタン)"
                          sx={{ borderRadius: 2 }}
                          multiline
                          rows={2}
                        />
                        <Button
                          variant="contained"
                          onClick={sendChatMessage}
                          sx={{ borderRadius: 3, px: 2, py: 1, fontWeight: 'bold', alignSelf: 'end' }}
                        >
                          送信
                        </Button>
                      </Box>
                    </Box>
                  </Paper>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </Box>
        <Dialog open={showSaveModal} onClose={() => setShowSaveModal(false)} fullWidth maxWidth="md">
          <DialogTitle>この内容で保存します</DialogTitle>
          <DialogContent dividers>
            <DialogContentText sx={{ whiteSpace: "pre-wrap" }}>
              <strong>タイトル:</strong> {title}
            </DialogContentText>
            <DialogContentText sx={{ whiteSpace: "pre-wrap" }}>
              <strong>文字起こし:</strong> {notes}
            </DialogContentText>
            <DialogContentText sx={{ whiteSpace: "pre-wrap" }}>
              <strong>議事録・改善案:</strong> {summary}
            </DialogContentText>
          </DialogContent>
          <DialogActions sx={{ justifyContent: 'center' }}>
            <Button onClick={saveToDatabase} variant="contained" sx={{ borderRadius: 3, px: 4, py: 1, fontWeight: 'bold' }}>
              この内容で保存する
            </Button>
            <Button onClick={() => setShowSaveModal(false)} variant="outlined" sx={{ borderRadius: 3, px: 4, py: 1, fontWeight: 'bold' }}>
              閉じる
            </Button>
          </DialogActions>
        </Dialog>

        <Dialog open={loading} maxWidth="xs">
          <DialogContent sx={{ textAlign: 'center' }}>
            <CircularProgress />
            <DialogContentText sx={{ mt: 2 }}>
              議事録を生成中です...お待ちください
            </DialogContentText>
          </DialogContent>
        </Dialog>
        <Dialog open={Boolean(saveStatus)} onClose={() => setSaveStatus(null)} maxWidth="xs">
          <DialogContent>
            <DialogContentText>{saveStatus}</DialogContentText>
          </DialogContent>
          <DialogActions sx={{ justifyContent: 'center' }}>
            <Button onClick={() => setSaveStatus(null)} variant="contained" sx={{ borderRadius: 3, px: 4, py: 1, fontWeight: 'bold' }}>
              閉じる
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </ThemeProvider>
  );
}

export default App;
