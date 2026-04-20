const baseUrl = 'http://13.235.78.53:80';
const API_KEY_STORAGE_KEY = 'youtube_api_key';

// State management
let state = {
  posComments: [],
  negComments: [],
  neuComments: [],
  isLoading: false
};

function showStatus(message, type = '') {
  const statusEl = document.getElementById('status');
  const icon = type === 'error' ? '?' : type === 'success' ? '?' : '?';
  statusEl.innerHTML = '<span class="icon">' + icon + '</span> ' + message;
  statusEl.className = 'status ' + type;
  statusEl.classList.remove('hidden');
}

function hideStatus() {
  document.getElementById('status').classList.add('hidden');
}

function updateProgress(percent, message) {
  const container = document.getElementById('progress-container');
  const fill = document.getElementById('progress-fill');
  const text = document.getElementById('progress-text');
  container.classList.add('visible');
  fill.style.width = percent + '%';
  text.textContent = message;
}

function hideProgress() {
  document.getElementById('progress-container').classList.remove('visible');
}

function setLoading(loading) {
  const btn = document.getElementById('fetch_comments');
  btn.disabled = loading;
  btn.classList.toggle('loading', loading);
  state.isLoading = loading;
}

function getApiKeyInput() {
  return document.getElementById('youtube_api_key');
}

function getStoredApiKey() {
  return localStorage.getItem(API_KEY_STORAGE_KEY) || '';
}

function setStoredApiKey(apiKey) {
  localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

async function getTotalComments(videoId, apiKey) {
  const response = await fetch(baseUrl + '/fetch-youtube-comments', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_id: videoId, api_key: apiKey })
  });

  if (!response.ok) {
    let errorMessage = 'Failed to fetch comments from backend';
    try {
      const errorData = await response.json();
      if (errorData.detail) errorMessage = errorData.detail;
    } catch (_) {
      // Keep the fallback message when the backend response is not JSON.
    }
    throw new Error(errorMessage);
  }

  const data = await response.json();
  return { totalComments: data.total_comments, allComments: data.comments };
}

async function getPredictions(comments) {
  const response = await fetch(baseUrl + '/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: comments })
  });

  const data = await response.json();
  let negative = 0;
  let neutral = 0;
  let positive = 0;

  data.predictions.forEach(p => {
    if (p === -1) negative++;
    else if (p === 0) neutral++;
    else if (p === 1) positive++;
  });

  return { allPredictions: data.predictions, pos: positive, neg: negative, neu: neutral };
}

async function fetchPieChart(pos, neu, neg) {
  try {
    const response = await fetch(baseUrl + '/pie-chart', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ positive: pos, neutral: neu, negative: neg })
    });

    if (!response.ok) throw new Error('Chart API error');

    const blob = await response.blob();
    const imgURL = URL.createObjectURL(blob);
    const graphDiv = document.getElementById('sentiment-graph');
    graphDiv.innerHTML = '';
    const img = document.createElement('img');
    img.src = imgURL;
    img.style.width = '100%';
    img.style.borderRadius = '8px';
    graphDiv.appendChild(img);
  } catch (error) {
    console.error('Error:', error);
    showStatus('Failed to load chart', 'error');
  }
}

function renderComments() {
  const renderTab = (comments, containerId) => {
    const container = document.getElementById(containerId);
    if (comments.length === 0) {
      container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:20px;">No comments</p>';
      return;
    }
    container.innerHTML = comments
      .map(comment => '<div class="comment-item">' + escapeHtml(comment) + '</div>')
      .join('');
  };

  renderTab(state.posComments, 'tab-content-positive');
  renderTab(state.negComments, 'tab-content-negative');
  renderTab(state.neuComments, 'tab-content-neutral');
}

function switchTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active', 'positive', 'negative', 'neutral');
    if (btn.dataset.tab === tabName) btn.classList.add('active', tabName);
  });

  document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
  document.getElementById('tab-content-' + tabName).classList.remove('hidden');
}

function updateSentimentDisplay(positive, negative, neutral, total) {
  const posPercent = total > 0 ? Math.round((positive / total) * 100) : 0;
  const negPercent = total > 0 ? Math.round((negative / total) * 100) : 0;
  const neuPercent = total > 0 ? Math.round((neutral / total) * 100) : 0;

  document.getElementById('positive-percent').textContent = posPercent + '%';
  document.getElementById('negative-percent').textContent = negPercent + '%';
  document.getElementById('neutral-percent').textContent = neuPercent + '%';

  document.getElementById('positive-count').textContent = positive + ' comments';
  document.getElementById('negative-count').textContent = negative + ' comments';
  document.getElementById('neutral-count').textContent = neutral + ' comments';
}

document.getElementById('fetch_comments').addEventListener('click', async () => {
  if (state.isLoading) return;

  const apiKey = getApiKeyInput().value.trim();
  if (!apiKey) {
    showStatus('Enter your YouTube API key before fetching comments', 'error');
    getApiKeyInput().focus();
    return;
  }

  setStoredApiKey(apiKey);

  chrome.tabs.query({ active: true, currentWindow: true }, async tabs => {
    const url = tabs[0].url;
    const match = url.match(/^(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})/);

    if (!match) {
      showStatus('Please open a YouTube video', 'error');
      return;
    }

    const videoId = match[1];

    document.getElementById('video-title').textContent = 'Analyzing video...';
    document.getElementById('video-title').classList.remove('placeholder');

    setLoading(true);
    hideStatus();

    try {
      updateProgress(20, 'Fetching comments...');
      const commentsData = await getTotalComments(videoId, apiKey);
      const allComments = commentsData.allComments;

      if (allComments.length === 0) {
        showStatus('No comments found', 'error');
        setLoading(false);
        hideProgress();
        return;
      }

      updateProgress(60, 'Analyzing sentiment...');
      const predictions = await getPredictions(allComments);
      updateProgress(90, 'Generating visualization...');

      state.posComments = allComments.filter((_, i) => predictions.allPredictions[i] === 1);
      state.negComments = allComments.filter((_, i) => predictions.allPredictions[i] === -1);
      state.neuComments = allComments.filter((_, i) => predictions.allPredictions[i] === 0);

      updateSentimentDisplay(predictions.pos, predictions.neg, predictions.neu, allComments.length);
      document.getElementById('sentiment-container').classList.remove('hidden');
      await fetchPieChart(predictions.pos, predictions.neu, predictions.neg);
      renderComments();
      document.getElementById('tabs-container').classList.add('visible');

      showStatus('Analyzed ' + allComments.length + ' comments', 'success');
      setTimeout(() => {
        hideProgress();
        setLoading(false);
      }, 1000);
    } catch (error) {
      showStatus('Error: ' + error.message, 'error');
      setLoading(false);
      hideProgress();
    }
  });
});

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

document.querySelectorAll('.box').forEach(box => {
  box.addEventListener('click', () => {
    switchTab(box.dataset.tab);
    document.getElementById('tabs-container').scrollIntoView({ behavior: 'smooth' });
  });
});

chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
  if (tabs[0] && tabs[0].url && tabs[0].url.match(/youtube\.com\/watch\?v=/)) {
    document.getElementById('video-title').textContent = 'YouTube video detected';
    document.getElementById('video-title').classList.remove('placeholder');
  }
});

document.addEventListener('DOMContentLoaded', () => {
  getApiKeyInput().value = getStoredApiKey();
});
