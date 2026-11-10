# TEMPORAL: capturas del visor en un Chrome real (se borra antes del PR)
import asyncio, sys
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path='/usr/bin/google-chrome')
        for nombre, url, acciones, espera in [
            ('satelite', 'http://127.0.0.1:8000/satelite.html', [], 70000),
            ('satelite_polvo', 'http://127.0.0.1:8000/satelite.html', ["document.querySelectorAll('input[name=ct-canal]')[5].click()"], 70000),
            ('satelite_wv', 'http://127.0.0.1:8000/satelite.html', ["document.querySelectorAll('input[name=ct-canal]')[3].click()"], 70000),
            ('trayectorias', 'http://127.0.0.1:8000/trayectorias.html', [], 8000),
            ('trayectorias_h48', 'http://127.0.0.1:8000/trayectorias.html',
             ["(()=>{const s=document.querySelector('[data-k=hour]');s.value=48;s.oninput({target:s})})()"], 4000),
            ('index', 'http://127.0.0.1:8000/index.html', [], 8000),
        ]:
            pg = await b.new_page(viewport={'width': 1400, 'height': 1000})
            errs = []
            pg.on('pageerror', lambda e: errs.append(str(e)))
            pg.on('console', lambda m: m.type == 'error' and errs.append(m.text))
            await pg.goto(url, wait_until='domcontentloaded', timeout=60000)
            await pg.wait_for_timeout(4000)
            for a in acciones:
                await pg.evaluate(a)
            await pg.wait_for_timeout(espera)
            if nombre.startswith('satelite'):
                await pg.evaluate("document.querySelector('[data-k=play]').click()")  # pausa
                await pg.evaluate("(()=>{const s=document.querySelector('[data-k=frame]');s.value=s.max;s.oninput({target:s})})()")
                await pg.wait_for_timeout(1500)
                print(nombre, await pg.evaluate("document.querySelector('[data-k=tlabel]').innerText"))
            await pg.screenshot(path=f'/tmp/capturas/{nombre}.png', full_page=True)
            print(nombre, 'errores:', errs[:5])
            await pg.close()
        await b.close()

asyncio.run(main())
