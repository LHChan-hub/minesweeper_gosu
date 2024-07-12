document.addEventListener("DOMContentLoaded", function() {
    const board = document.getElementById('board');
    for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 10; j++) {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            cell.dataset.x = i;
            cell.dataset.y = j;
            cell.addEventListener('click', clickCell);
            board.appendChild(cell);
        }
    }
});

function clickCell(event) {
    const cell = event.target;
    const x = cell.dataset.x;
    const y = cell.dataset.y;
    fetch('/click_cell', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ x: x, y: y })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'mine') {
            cell.style.backgroundColor = 'red';
            alert('지뢰를 찾았습니다!');
        } else {
            cell.style.backgroundColor = 'white';
        }
    });
}

function agentPlay() {
    const state = getCurrentState();
    fetch('/agent_play', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ state: state })
    })
    .then(response => response.json())
    .then(data => {
        const action = data.action;
        const x = Math.floor(action / 10);
        const y = action % 10;
        const cell = document.querySelector(`.cell[data-x="${x}"][data-y="${y}"]`);
        cell.click();
    });
}

function getCurrentState() {
    // 현재 게임 상태를 배열로 반환
    return Array.from(document.querySelectorAll('.cell')).map(cell => {
        return cell.style.backgroundColor === 'red' ? -1 : (cell.style.backgroundColor === 'white' ? 1 : 0);
    });
}
