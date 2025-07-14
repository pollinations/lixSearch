
marked.setOptions({
    breaks: true, 
    gfm: true,    
    headerIds: false,
    headerPrefix: '',
    smartLists: true,
    smartypants: true,
    xhtml: false,
    highlight: function(code, lang) {
        return `<code class="language-${lang || 'text'}">${code}</code>`;
    }
});

// const SERVER_URL = "http://127.0.0.1:5000";
const SERVER_URL = "https://search.pollinations.ai";
document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();
});

const input = document.getElementById('queryInput');
const submitButton = document.getElementById('submitButton');
const sseFeed = document.getElementById('sseFeed');
const finalOutput = document.getElementById('finalOutput');
const sseToggle = document.getElementById('sseToggle');
const classicBtn = document.getElementById('classicBtn');
const chatBtn = document.getElementById('chatBtn');
let useChatApi = false;

function clearOutputs() {
    sseFeed.innerHTML = '';
    finalOutput.innerHTML = '';
    sseFeed.style.display = sseToggle.checked ? 'block' : 'none';
}

classicBtn.addEventListener('click', () => {
    useChatApi = false;
    classicBtn.classList.add('active');
    chatBtn.classList.remove('active');
    clearOutputs();
});

chatBtn.addEventListener('click', () => {
    useChatApi = true;
    chatBtn.classList.add('active');
    classicBtn.classList.remove('active');
    clearOutputs();
});

function showResult(label, data) {
    finalOutput.innerHTML = '';
    let div = document.createElement('div');
    if (typeof data === 'object' && data !== null && data.error) {
        div.innerText = `[${label}] ERROR: ${data.error}`;
        div.className = 'text-red-400';
    } else if (typeof data === 'object' && data !== null && data.result !== undefined) {
        div.innerHTML = marked.parse(data.result);
    } else {
        div.innerText = `[${label}] ${typeof data === 'string' ? data : JSON.stringify(data, null, 2)}`;
    }
    finalOutput.appendChild(div);
}

function doSearch() {
    clearOutputs();

    if (sseToggle.checked) {
        sseFeed.style.display = 'block';
        const finalChunks = [];

        fetch(`${SERVER_URL}/search/sse`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: input.value })
        }).then(response => {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            function readStream() {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        if (finalChunks.length > 0) {
                            renderFinalOutput(finalChunks.join(''));
                        }
                        return;
                    }
                    buffer += decoder.decode(value, { stream: true });
                    let parts = buffer.split('\n\n');
                    buffer = parts.pop();
                    parts.forEach(handleSSEPart);
                    readStream();
                });
            }

            function handleSSEPart(part) {
                let lines = part.trim().split('\n');
                let event = '', data = '';
                for (let line of lines) {
                    if (line.startsWith('event:')) event = line.replace('event:', '').trim();
                    if (line.startsWith('data:')) data += line.replace('data:', '').trim() + '\n';
                }
                data = data.trim();

                if (event === 'final-part') {
                    finalChunks.push(data);
                } else if (event === 'final') {
                    finalChunks.push(data);
                    renderFinalOutput(finalChunks.join(''));
                } else if (event === 'error') {
                    finalOutput.innerHTML = `<span class="text-red-400">${data}</span>`;
                } else if (event) {
                    const div = document.createElement('div');
                    div.className = 'mb-1';
                    div.innerHTML = `<span class="text-blue-400 font-mono">[${event}]</span> ${data}`;
                    sseFeed.appendChild(div);
                    sseFeed.scrollTop = sseFeed.scrollHeight;
                }
            }

            function renderFinalOutput(markdown) {
                let html = marked.parse(markdown);
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = html;

                const images = Array.from(tempDiv.querySelectorAll('img'));
                if (images.length > 0) {
                    const imageGrid = document.createElement('div');
                    imageGrid.className = 'image-grid'; // CSS handles `overflow-x-auto flex-nowrap`
                    images.forEach(img => imageGrid.appendChild(img));
                    // Remove images from their original parsed positions
                    tempDiv.querySelectorAll('img').forEach(img => img.remove()); 
                    // Insert the image grid at the beginning of the content or a suitable place
                    // For simplicity, let's append it to the tempDiv.
                    // If you want images to appear *before* text, you'd insertBefore.
                    tempDiv.prepend(imageGrid); 
                }

                finalOutput.innerHTML = tempDiv.innerHTML;
            }

            readStream();
        });
    } 

    else if (useChatApi) {
        finalOutput.innerHTML = '<span class="text-gray-400">Loading (Chat API)...</span>';
        fetch(`${SERVER_URL}/search/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: [{ role: "user", content: input.value }]
            })
        })
        .then(r => r.json())
        .then(data => {
            if (data.choices && data.choices[0]) {
                showResult('Chat API', { result: data.choices[0].message.content });
            } else {
                showResult('Chat API', data);
            }
        })
        .catch(e => showResult('Chat API', { error: e.toString() }));
        return;
    }
    
    else {
        sseFeed.style.display = 'none';
        finalOutput.innerHTML = '<span class="text-gray-400">Loading...</span>';

        fetch(`${SERVER_URL}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: input.value })
        })
        .then(r => r.json())
        .then(data => showResult('POST', data))
        .catch(e => showResult('POST', { error: e.toString() }));
    }
}

submitButton.onclick = (e) => {
    e.preventDefault();
    doSearch();
};

sseToggle.onchange = () => {
    clearOutputs();
};

input.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        doSearch();
    }
});
