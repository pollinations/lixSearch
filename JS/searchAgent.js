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

const SERVER_URL = "http://127.0.0.1:5000";
const input = document.getElementById('queryInput');
const submitButton = document.getElementById('submitButton');
const sseFeed = document.getElementById('sseFeed');
const finalOutput = document.getElementById('finalOutput');
const sseToggle = document.getElementById('sseToggle');
const classicBtn = document.getElementById('classicBtn');
const chatBtn = document.getElementById('chatBtn');
const imageUrlInput = document.getElementById('imageUrlInput');
const statusMessage = document.getElementById('sseFeed');
const deepToggle = document.getElementById('deepToggle');

let useChatApi = false;
let hasReachedSuccess = false;

function clearOutputs() {
    sseFeed.innerHTML = '';
    finalOutput.innerHTML = '';
    statusMessage.innerHTML = '';
    statusMessage.classList.add('hidden');
    sseFeed.style.display = sseToggle.checked ? 'flex' : 'none';
}

function updateStatus(msg, type = "info") {
    let msgNode = `<span> <span class="information_type"> [INFO] </span> ${msg}  </span>`;
    statusMessage.innerHTML += msgNode;
    statusMessage.classList.remove('hidden');
    statusMessage.className = `text-center mb-6 ${type === "error" ? "text-red-400" : "text-blue-400"}`;
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
    } else if (typeof data === 'object' && data.choices && data.choices[0]) {
        div.innerHTML = marked.parse(data.choices[0].message.content);
    } else {
        div.innerText = `[${label}] ${typeof data === 'string' ? data : JSON.stringify(data, null, 2)}`;
    }
    finalOutput.appendChild(div);
}

function renderFinalOutput(markdown) {
    let html = marked.parse(markdown);
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = html;
    const images = Array.from(tempDiv.querySelectorAll('img'));
    if (images.length > 0) {
        const imageGrid = document.createElement('div');
        imageGrid.className = 'image-grid';
        images.forEach(img => imageGrid.appendChild(img));
        tempDiv.querySelectorAll('img').forEach(img => img.remove());
        tempDiv.append(imageGrid);
    }
    finalOutput.innerHTML = tempDiv.innerHTML;
}

function buildPayload(query, imageUrl) {
    let content = [];
    if (query) content.push({ type: "text", text: query });
    if (imageUrl) content.push({ type: "image_url", image_url: { url: imageUrl } });
    return {
        stream: sseToggle.checked,
        messages: [{ role: "user", content }],
        deep: deepToggle.checked // <-- Add this line
    };
}

function doSearch() {
    clearOutputs();
    const query = input.value;
    const imageUrl = imageUrlInput.value.trim();
    input.value = '';
    imageUrlInput.value = '';

    let payload = buildPayload(query, imageUrl);

    if (sseToggle.checked) {
        sseFeed.style.display = 'flex';
        let taskChunks = [];
        let finalMarkdown = '';
        let hasSuccess = false;
        let hasError = false;

        fetch(`${SERVER_URL}/search/sse`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(response => {
            if (!response.ok) throw new Error("Network error");
            const reader = response.body.getReader();
            let decoder = new TextDecoder();
            let buffer = '';

            function processChunk({ done, value }) {
                if (done) return;
                buffer += decoder.decode(value, { stream: true });
                let lines = buffer.split('\n');
                buffer = lines.pop();

                lines.forEach(line => {
                    if (line.startsWith('data:')) {
                        let data = line.slice(5).trim();
                        if (data === '[DONE]') {
                            esClose();
                        } else {
                            try {
                                const parsed = JSON.parse(data);
                                if (parsed.error) {
                                    updateStatus(parsed.error, 'error');
                                    hasError = true;
                                }
                                const delta = parsed?.choices?.[0]?.delta?.content;
                                if (delta) {
                                    // Check for <TASK>...</TASK> and SUCCESS
                                    if (!hasSuccess) {
                                        // Extract all <TASK>...</TASK> blocks
                                        const taskMatches = [...delta.matchAll(/<TASK>([\s\S]*?)<\/TASK>/g)];
                                        taskMatches.forEach(match => {
                                            taskChunks.push(match[1]);
                                            // Optionally, show in sseFeed
                                            const div = document.createElement('div');
                                            div.innerText = match[1];
                                            sseFeed.appendChild(div);
                                        });
                                        // Check for SUCCESS marker
                                        if (delta.includes('SUCCESS')) {
                                            hasSuccess = true;
                                            // Everything after SUCCESS is final markdown
                                            const successIndex = delta.indexOf('SUCCESS');
                                            finalMarkdown += delta.slice(successIndex + 'SUCCESS'.length);
                                        }
                                    } else {
                                        // After SUCCESS, accumulate for final markdown
                                        finalMarkdown += delta;
                                    }
                                }
                            } catch (e) {
                                updateStatus(data, 'info');
                            }
                        }
                    }
                });

                return reader.read().then(processChunk);
            }

            function esClose() {
                statusMessage.classList.add('hidden');
                if (finalMarkdown.trim() && !hasError) {
                    renderFinalOutput(finalMarkdown);
                }
            }

            return reader.read().then(processChunk);
        })
        .catch(e => {
            updateStatus(`[ERROR] ${e.toString()}`, "error");
            finalOutput.innerHTML = `<span class="text-red-400">[ERROR] ${e.toString()}</span>`;
        });
    } else {
        sseFeed.style.display = 'none';
        finalOutput.innerHTML = '<span class="text-gray-400">Loading...</span>';
        fetch(`${SERVER_URL}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
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
