const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  const domains = new Set();

  page.on('request', req => {
    try {
      domains.add(new URL(req.url()).hostname);
    } catch {}
  });

  await page.goto('https://drocheam.github.io/optrace/details/sampling.html');

  await page.waitForTimeout(3000);

  console.log([...domains].sort().join('\n'));

  await browser.close();
})();

