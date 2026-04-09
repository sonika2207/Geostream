/**
 * GeoStream — Frontend Logic
 * Handles form submission, pipeline status polling, loading states
 */

// ── DOM Refs ──
const form = document.getElementById('generateForm');
const submitBtn = document.getElementById('submitBtn');
const btnText = submitBtn?.querySelector('.btn-text');
const btnLoader = submitBtn?.querySelector('.btn-loader');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingStage = document.getElementById('loadingStage');
const progressBar = document.getElementById('progressBar');
const loadingMessage = document.getElementById('loadingMessage');

// ── Particles ──
(function initParticles() {
    const container = document.getElementById('particles');
    if (!container) return;
    for (let i = 0; i < 40; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        p.style.left = Math.random() * 100 + '%';
        p.style.animationDelay = Math.random() * 20 + 's';
        p.style.animationDuration = (15 + Math.random() * 25) + 's';
        const size = 2 + Math.random() * 4;
        p.style.width = p.style.height = size + 'px';
        container.appendChild(p);
    }
})();

// ── Form Submit ──
if (form) {
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const date = document.getElementById('date').value;
        const fromTime = document.getElementById('from_time').value;
        const toTime = document.getElementById('to_time').value;
        const enhance = document.getElementById('enhance') ? document.getElementById('enhance').checked : false;

        // Validation
        if (!date || !fromTime || !toTime) {
            showToast('Please fill in all fields', 'warning');
            return;
        }

        if (fromTime >= toTime) {
            showToast('Start time must be before end time', 'warning');
            return;
        }

        // Show loading
        setLoading(true);

        // Reset stage text just in case
        if (loadingStage) loadingStage.textContent = 'Initializing...';
        if (loadingMessage) loadingMessage.textContent = 'Starting pipeline...';
        if (progressBar) progressBar.style.width = '0%';

        try {
            const resp = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    date,
                    from_time: fromTime,
                    to_time: toTime,
                    enhance,
                    rife_exp: 1
                })
            });

            const data = await resp.json();

            if (!resp.ok) {
                showToast(data.error || 'Failed to start pipeline', 'error');
                setLoading(false);
                return;
            }

            // Start polling
            pollStatus();
        } catch (err) {
            showToast('Server error: ' + err.message, 'error');
            setLoading(false);
        }
    });
}

// ── Status Polling ──
let pollInterval = null;

function pollStatus() {
    pollInterval = setInterval(async () => {
        try {
            const resp = await fetch('/status');
            const s = await resp.json();

            updateProgress(s);

            if (s.stage === 'done') {
                clearInterval(pollInterval);
                // Short delay before redirect for UX
                setTimeout(() => {
                    window.location.href = '/result.html';
                }, 800);
            }

            if (s.stage === 'error') {
                clearInterval(pollInterval);
                showToast(s.message || 'Pipeline failed', 'error');
                setLoading(false);
            }
        } catch (err) {
            // Ignore transient fetch errors
        }
    }, 1500);
}

function updateProgress(s) {
    const stageNames = {
        fetching: '📡 Fetching Satellite Frames',
        interpolating: '🔀 RIFE Interpolation',
        enhancing: '✨ Real-ESRGAN Enhancement',
        encoding: '🎬 Encoding Video',
        done: '✅ Complete!',
        error: '❌ Error',
    };

    if (loadingStage) loadingStage.textContent = stageNames[s.stage] || s.stage;
    if (progressBar) progressBar.style.width = s.progress + '%';
    if (loadingMessage) loadingMessage.textContent = s.message || '';
}

// ── UI Helpers ──
function setLoading(on) {
    if (loadingOverlay) loadingOverlay.style.display = on ? 'flex' : 'none';
    if (btnText) btnText.style.display = on ? 'none' : 'inline';
    if (btnLoader) btnLoader.style.display = on ? 'inline-flex' : 'none';
    if (submitBtn) submitBtn.disabled = on;
}

function showToast(msg, type = 'info') {
    // Remove existing toasts
    document.querySelectorAll('.toast').forEach(t => t.remove());

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.style.cssText = `
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        padding: 14px 28px;
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
        z-index: 200;
        animation: toast-in 0.4s ease;
        color: white;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    `;

    const colors = {
        info: 'rgba(99, 102, 241, 0.9)',
        warning: 'rgba(245, 158, 11, 0.9)',
        error: 'rgba(239, 68, 68, 0.9)',
        success: 'rgba(34, 197, 94, 0.9)',
    };
    toast.style.background = colors[type] || colors.info;
    toast.textContent = msg;

    document.body.appendChild(toast);

    // Auto remove
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Add toast animation
const style = document.createElement('style');
style.textContent = `
    @keyframes toast-in {
        from { opacity: 0; transform: translateX(-50%) translateY(20px); }
        to   { opacity: 1; transform: translateX(-50%) translateY(0); }
    }
`;
document.head.appendChild(style);
