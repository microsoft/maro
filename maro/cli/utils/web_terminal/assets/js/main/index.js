$(document).ready(function () {
  // Load terminal addons.
  Terminal.applyAddon(fullscreen)
  Terminal.applyAddon(fit)
  Terminal.applyAddon(webLinks)
  Terminal.applyAddon(search)

  // Create xterms.js terminal.
  const waitMS = 50
  const term = new Terminal({
    cols: 1,
    rows: 15,
    cursorBlink: true,
    macOptionIsMeta: true,
    scrollback: true
  })

  term.open(document.getElementById('terminal'))
  term.fit()
  // term.resize(120, 20);
  console.log(`size: ${term.cols} columns, ${term.rows} rows`)
  term.fit()
  term.on('key', (key, ev) => {
    socket.emit('pty-input', { input: key })
  })

  term.prompt = () => {
    term.write('$')
  }

  // Create socket.io connection.
  const socket = io.connect('/pty')

  socket.on('pty-output', function (data) {
    console.log('new output', data)
    term.write(data.output)
  })

  socket.on('connect', () => {
    console.log('socket.io connected.')
  })

  socket.on('disconnect', () => {
    console.log('socket.io disconnected.')
  })

  function fitToScreen () {
    term.fit()
    socket.emit('resize', { cols: term.cols, rows: term.rows })
  }

  function terminalResize (func, waitMS) {
    let timeout
    return function (...args) {
      const context = this
      clearTimeout(timeout)
      timeout = setTimeout(() => func.apply(context, args), waitMS)
    }
  }

  window.onresize = terminalResize(fitToScreen, waitMS)
  term.write('Welcome to MARO.\r\n')
  term.write('Repository: https://github.com/microsoft/maro\r\n')
  term.write('Documentation: https://maro.readthedocs.io/en/latest/\r\n')
  term.prompt()
})
