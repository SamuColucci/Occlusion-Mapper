import os
import csv
import webbrowser

def generate_html_report(csv_fpath="report_storico_bayes.csv", html_fpath="visualizzatore_report.html"):
    if not os.path.exists(csv_fpath):
        print(f"[ERROR] Il file '{csv_fpath}' non esiste. Avvia prima l'agente Bayesiano.")
        return False
        
    rows = []
    try:
        with open(csv_fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception as e:
        print(f"[ERROR] Impossibile leggere il file CSV: {e}")
        return False
        
    # Helper per colorare le singole celle di probabilità nel report
    def get_prob_cell(val_str):
        try:
            val = float(val_str)
        except Exception:
            val = 0.0
        percent = val * 100
        
        # Colore di sfondo e testo in base alla fascia di probabilità
        if val >= 0.90:
            style = "background-color: rgba(255, 23, 68, 0.22); color: #ff3d57; font-weight: 600;"
        elif val >= 0.20:
            style = "background-color: rgba(255, 214, 0, 0.12); color: #ffeb3b; font-weight: 500;"
        elif val > 0.01:
            style = "background-color: rgba(0, 230, 118, 0.06); color: #00e676;"
        else:
            style = "color: rgba(255, 255, 255, 0.25); font-size: 0.8rem;"
            
        return f'<td style="{style}">{percent:.1f}%</td>'

    # Costruiamo le righe della tabella in HTML
    table_rows_html = ""
    for idx, r in enumerate(rows):
        # Valutazione probabilità massima per il badge complessivo
        prob_val = float(r.get("Prob_Massima", 0.0))
        prob_percent = prob_val * 100
        if prob_percent >= 90:
            badge_class = "badge-danger"
        elif prob_percent >= 20:
            badge_class = "badge-warning"
        else:
            badge_class = "badge-success"
            
        # Colonne probabilistiche per ciascuna delle 10 classi
        prob_tds = ""
        for col_name in ["Prob_Auto", "Prob_Pedone", "Prob_Camion", "Prob_Bicicletta", "Prob_Moto", 
                         "Prob_Bus", "Prob_Rimorchio", "Prob_Barriera", "Prob_Cono", "Prob_Altro"]:
            prob_tds += get_prob_cell(r.get(col_name, "0.0"))
            
        sintesi = r.get("Sintesi", "")
        
        table_rows_html += f"""
        <tr>
            <td class="font-mono">{r.get("Scena")}</td>
            <td>{r.get("Frame_Num")}</td>
            <td class="font-mono text-muted" title="{r.get("Lidar_Token")}">{r.get("Lidar_Token")[:6]}...</td>
            <td><span class="occ-idx">{r.get("Occlusion_Index")}</span></td>
            <td>{r.get("Area_m2")} m²</td>
            <td>{r.get("Distanza_m")}m</td>
            {prob_tds}
            <td class="font-semibold text-accent">{r.get("Oggetto_Piu_Probabile")}</td>
            <td><span class="badge {badge_class}">{prob_percent:.1f}%</span></td>
            <td class="text-left font-italic text-muted">{sintesi}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Storico Stime Bayesiane - Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-color: #f1f5f9;
            --text-muted: #94a3b8;
            --accent-color: #00f0ff;
            --danger-color: #ff1744;
            --warning-color: #ffd600;
            --success-color: #00e676;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            padding: 30px 15px;
            line-height: 1.5;
            background-image: radial-gradient(circle at 10% 20%, rgba(0, 240, 255, 0.05) 0%, transparent 40%),
                              radial-gradient(circle at 90% 80%, rgba(255, 23, 68, 0.03) 0%, transparent 40%);
            background-attachment: fixed;
        }}
        
        .container {{
            max-width: 1750px;
            margin: 0 auto;
        }}
        
        header {{
            margin-bottom: 25px;
            text-align: center;
        }}
        
        h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(to right, #fff, var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        }}
        
        .subtitle {{
            color: var(--text-muted);
            font-size: 0.95rem;
        }}
        
        .controls {{
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            justify-content: space-between;
            align-items: center;
        }}
        
        .search-box {{
            flex: 1;
            position: relative;
        }}
        
        .search-box input {{
            width: 100%;
            padding: 12px 20px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-color);
            font-family: inherit;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }}
        
        .search-box input:focus {{
            border-color: var(--accent-color);
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.2);
        }}
        
        .stats-badge {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 0.9rem;
            color: var(--text-muted);
            backdrop-filter: blur(10px);
        }}
        
        .stats-badge strong {{
            color: var(--accent-color);
        }}
        
        .table-container {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: center;
            font-size: 0.85rem;
        }}
        
        th {{
            background-color: rgba(15, 23, 42, 0.85);
            color: var(--text-muted);
            font-weight: 600;
            padding: 12px 8px;
            border-bottom: 1px solid var(--border-color);
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.8px;
        }}
        
        td {{
            padding: 10px 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            transition: background-color 0.2s ease;
        }}
        
        tr:hover td {{
            background-color: rgba(255, 255, 255, 0.015);
        }}
        
        .font-mono {{
            font-family: 'Fira Code', monospace;
            font-size: 0.8rem;
        }}
        
        .font-semibold {{
            font-weight: 600;
        }}
        
        .font-italic {{
            font-style: italic;
        }}
        
        .text-left {{
            text-align: left;
        }}
        
        .text-accent {{
            color: var(--accent-color);
        }}
        
        .text-muted {{
            color: var(--text-muted);
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .badge-danger {{
            background-color: rgba(255, 23, 68, 0.15);
            color: var(--danger-color);
            border: 1px solid rgba(255, 23, 68, 0.3);
        }}
        
        .badge-warning {{
            background-color: rgba(255, 214, 0, 0.1);
            color: var(--warning-color);
            border: 1px solid rgba(255, 214, 0, 0.3);
        }}
        
        .badge-success {{
            background-color: rgba(0, 230, 118, 0.1);
            color: var(--success-color);
            border: 1px solid rgba(0, 230, 118, 0.3);
        }}
        
        .occ-idx {{
            background: rgba(255, 255, 255, 0.04);
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Cronologia Storica Stime Bayesiane</h1>
            <p class="subtitle">Vettore completo di probabilità e dettagli per ogni area di occlusione del dataset</p>
        </header>
        
        <div class="controls">
            <div class="search-box">
                <input type="text" id="searchInput" placeholder="Cerca per scena, classe di oggetto, LiDAR Token o sintesi descrittiva..." onkeyup="filterTable()">
            </div>
            <div class="stats-badge">
                Righe visibili: <strong id="totalRows">{len(rows)}</strong>
            </div>
        </div>
        
        <div class="table-container">
            <table id="reportTable">
                <thead>
                    <tr>
                        <th>Scena</th>
                        <th>Frame</th>
                        <th>LiDAR</th>
                        <th>Occ</th>
                        <th>Area</th>
                        <th>Dist</th>
                        <!-- Le 10 colonne probabilistiche complete del CSV -->
                        <th>Auto</th>
                        <th>Pedone</th>
                        <th>Camion</th>
                        <th>Bici</th>
                        <th>Moto</th>
                        <th>Bus</th>
                        <th>Rimorchio</th>
                        <th>Barriera</th>
                        <th>Cono</th>
                        <th>Altro</th>
                        <th>Best Class</th>
                        <th>Max Prob</th>
                        <th class="text-left" style="min-width: 250px;">Sintesi Descrittiva Storico</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows_html}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        function filterTable() {{
            var input = document.getElementById("searchInput");
            var filter = input.value.toUpperCase();
            var table = document.getElementById("reportTable");
            var tr = table.getElementsByTagName("tr");
            var visibleCount = 0;
            
            for (var i = 1; i < tr.length; i++) {{
                var showRow = false;
                var tds = tr[i].getElementsByTagName("td");
                for (var j = 0; j < tds.length; j++) {{
                    if (tds[j]) {{
                        var textValue = tds[j].textContent || tds[j].innerText;
                        if (textValue.toUpperCase().indexOf(filter) > -1) {{
                            showRow = true;
                            break;
                        }}
                    }}
                }}
                if (showRow) {{
                    tr[i].style.display = "";
                    visibleCount++;
                }} else {{
                    tr[i].style.display = "none";
                }}
            }}
            document.getElementById("totalRows").innerText = visibleCount;
        }}
    </script>
</body>
</html>
"""
    try:
        with open(html_fpath, "w", encoding="utf-8") as f:
            f.write(html_content)
        return True
    except Exception as e:
        print(f"[ERROR] Impossibile scrivere il file HTML: {e}")
        return False

if __name__ == "__main__":
    success = generate_html_report()
    if success:
        print("[SUCCESS] Report HTML generato con successo! Apertura nel browser in corso...")
        webbrowser.open("visualizzatore_report.html")
    else:
        print("[ERROR] Generazione del report fallita.")
