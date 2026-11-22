// Recorre en Netron (el visor estándar de redes neuronales) el grafo ONNX de la red YOLOv10n de DIMMIT, en una
// ventana de Chromium sobre la pantalla actual (la virtual de la grabación): abre la primera convolución para ver
// sus pesos, baja por todo el grafo hasta la salida y abre los nodos finales (TopK de 300 y output0).
// Uso: node scripts/recorrer_red_netron.js http://127.0.0.1:8081/ 1600 900
const { chromium } = require('playwright');

(async () => {
  const [url = 'http://127.0.0.1:8081/', w = '1600', h = '900'] = process.argv.slice(2);
  const W = parseInt(w, 10), H = parseInt(h, 10);
  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', `--window-size=${W},${H}`, '--window-position=0,0', '--kiosk', '--no-first-run', '--disable-infobars'],
  });
  const page = await browser.newPage({ viewport: null });
  await page.goto(url);
  await page.waitForTimeout(6000);                       // entrada: images 1×3×640×640
  console.log('[netron] entrada images 1×3×640×640 y primeras convoluciones');
  await page.locator('text=Conv').first().click();       // propiedades de la primera convolución
  console.log('[netron] propiedades de la primera Conv: kernel 3×3, stride 2, pesos 16×3×3×3');
  await page.waitForTimeout(6000);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(800);
  await page.mouse.click(59, H - 27);                    // alejar un nivel para ver más nodos
  await page.mouse.move(W / 2, H / 2);
  console.log('[netron] recorriendo el grafo completo: backbone -> neck -> cabeza v10Detect -> TopK');
  for (let i = 0; i < 420; i++) {                        // ~20.000 px de grafo en ~60 s
    await page.mouse.wheel(0, 48);
    await page.waitForTimeout(110);
  }
  await page.waitForTimeout(2500);
  for (const nodo of ['TopK', 'output0']) {
    try {
      await page.getByText(nodo, { exact: true }).last().click({ timeout: 3000 });
      console.log(`[netron] propiedades de ${nodo}`);
      await page.waitForTimeout(5000);
      await page.keyboard.press('Escape');
      await page.waitForTimeout(600);
    } catch (e) {
      console.log(`[netron] no se encontró ${nodo} en pantalla`);
    }
  }
  await browser.close();
})();
