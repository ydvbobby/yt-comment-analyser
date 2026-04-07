

// State management
let state = {
  posComments: [],
  negComments: [],
  neuComments: [],
  isLoading: false
};

function showStatus(message, type = '') {
  const statusEl = document.getElementById('status');
  const icon = type === 'error' ? '❌' : type === 'success' ? '✅' : '⏳';
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

async function getTotalComments(videoId, apikey) {
  let allComments = [];
  let pageToken = '';
  let totalFetched = 0;
  const maxComments = 500;

  while (true) {
    let apiUrl = 'https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=' + videoId + '&maxResults=100&key=' + apikey;
    if (pageToken) apiUrl += '&pageToken=' + pageToken;

    const response = await fetch(apiUrl);
    const data = await response.json();

    if (!data.items) break;

    data.items.forEach(item => {
      allComments.push(item.snippet.topLevelComment.snippet.textDisplay);
      totalFetched++;
    });

    updateProgress(Math.min((totalFetched / maxComments) * 50, 50), 'Fetched ' + totalFetched + ' comments...');

    if (data.nextPageToken && totalFetched < maxComments) {
      pageToken = data.nextPageToken;
    } else {
      break;
    }
  }

  return { totalComments: allComments.length, allComments: allComments };
}

async function getPredictions(comments) {
  const response = await fetch('http://127.0.0.1:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: comments })
  });

  const data = await response.json();
  let negative = 0, neutral = 0, positive = 0;
  
  data.predictions.forEach(p => {
    if (p === -1) negative++;
    else if (p === 0) neutral++;
    else if (p === 1) positive++;
  });

  return { allPredictions: data.predictions, pos: positive, neg: negative, neu: neutral };
}

async function fetchPieChart(pos, neu, neg) {
  try {
    const response = await fetch('http://127.0.0.1:8000/pie-chart', {
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
    container.innerHTML = comments.slice(0, 20).map(c => '<div class="comment-item">' + c + '</div>').join('');
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

  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const match = url.match(/^(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})/);

    if (!match) {
      showStatus('Please open a YouTube video', 'error');
      return;
    }

require('dotenv').config();

const apiKey = process.env.YOUTUBE_API_KEY;

console.log(apiKey);

    const videoId = match[1];
    const apikey = apiKey
    
    document.getElementById('video-title').textContent = 'Analyzing video...';
    document.getElementById('video-title').classList.remove('placeholder');

    setLoading(true);
    hideStatus();

    try {
      const commentsData = await getTotalComments(videoId, apikey);
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
      document.getElementById('tabs-container').classList.remove('hidden');

      showStatus('Analyzed ' + allComments.length + ' comments', 'success');
      setTimeout(() => { hideProgress(); setLoading(false); }, 1000);

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

chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  if (tabs[0] && tabs[0].url && tabs[0].url.match(/youtube\.com\/watch\?v=/)) {
    document.getElementById('video-title').textContent = 'YouTube video detected';
    document.getElementById('video-title').classList.remove('placeholder');
  }
});
