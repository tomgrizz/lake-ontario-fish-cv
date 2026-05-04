import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from collections import defaultdict

SPECIES = {0:'Chinook', 1:'Coho', 2:'Atlantic', 3:'Rainbow Trout', 4:'Brown Trout'}
COLORS  = {0:'#E84C3D', 1:'#3498DB', 2:'#2ECC71', 3:'#9B59B6', 4:'#F39C12'}
MONTHS  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

conn = sqlite3.connect('G:/Projects/model_outputs/run_20260428/tracks.sqlite')
rows = conn.execute(
    'SELECT video_id, predicted_class, n_frames FROM tracks WHERE n_frames >= 5'
).fetchall()
conn.close()

by_month = defaultdict(lambda: defaultdict(int))

for vid, cls, nframes in rows:
    norm = vid.replace('\\', '/')
    parts = norm.split('/')
    if len(parts) < 3:
        continue
    folder = parts[2]  # e.g. "11022020-11052020"
    if len(folder) >= 4 and folder[:2].isdigit() and folder[2:4].isdigit():
        try:
            month = int(folder[:2])
            if 1 <= month <= 12:
                by_month[month][cls] += 1
        except ValueError:
            pass

month_totals = [sum(by_month[m].values()) for m in range(1,13)]

fig, axes = plt.subplots(2, 1, figsize=(12, 9))

# --- Absolute counts ---
ax = axes[0]
bottom = np.zeros(12)
for cls in range(5):
    vals = [by_month[m][cls] for m in range(1,13)]
    ax.bar(range(1,13), vals, bottom=bottom, color=COLORS[cls],
           label=SPECIES[cls], alpha=0.85, edgecolor='white', linewidth=0.3)
    bottom += np.array(vals)
ax.set_xticks(range(1,13))
ax.set_xticklabels(MONTHS)
ax.set_ylabel('Tracks detected')
ax.set_title(
    f'Species detections by month  —  Ganaraska 2020–2022  '
    f'({len(rows):,} tracks, inference {162560/246075*100:.0f}% complete)'
)
ax.legend(loc='upper left', fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
ax.grid(axis='y', alpha=0.3)

# --- Proportional ---
ax2 = axes[1]
bottom2 = np.zeros(12)
for cls in range(5):
    vals_pct = [
        (by_month[m][cls] / month_totals[m-1] * 100) if month_totals[m-1] > 0 else 0
        for m in range(1,13)
    ]
    ax2.bar(range(1,13), vals_pct, bottom=bottom2, color=COLORS[cls],
            label=SPECIES[cls], alpha=0.85, edgecolor='white', linewidth=0.3)
    bottom2 += np.array(vals_pct)
ax2.set_xticks(range(1,13))
ax2.set_xticklabels(MONTHS)
ax2.set_ylabel('% of monthly detections')
ax2.set_ylim(0, 100)
ax2.set_title('Species composition by month (%)')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = 'G:/Projects/species_by_month.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')

print('\nMonthly track counts:')
header = f'{"Month":>6}  {"Total":>7}  ' + '  '.join(f'{SPECIES[c][:10]:>10}' for c in range(5))
print(header)
for m in range(1,13):
    total = month_totals[m-1]
    if total == 0:
        continue
    counts = '  '.join(f'{by_month[m][c]:>10,}' for c in range(5))
    print(f'{MONTHS[m-1]:>6}  {total:>7,}  {counts}')
