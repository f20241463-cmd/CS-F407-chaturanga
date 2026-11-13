W=$(dirname $0); cd $W
WB=""; [ -f net.wasm ] && WB=$(base64 -i net.wasm | tr -d '\n')
{ cat head.html body.html; echo '<script>'; cat weights.js; echo "window.CZ_WASM=\"$WB\";"; echo '</script><script>'; cat engine.js; echo '</script><script>'; cat netjs.js; echo '</script><script>'; cat app.js; echo '</script>'; } > chaturangzero.html
(printf '<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"></head><body>'; cat chaturangzero.html; printf '</body></html>') > local.html
ls -la chaturangzero.html
