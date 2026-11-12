"""Generate two TeX tables from historical JSON; optionally compile the article."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def make_tables(root):
    source = json.loads((root / 'results/metrics.json').read_text(encoding='utf-8'))
    metrics = source
    matrix = source['confusion_matrix']
    target = root / 'article/generated'
    target.mkdir(parents=True, exist_ok=True)
    def value(x):
        return ('%.3f' % x).replace('.', ',')
    endrow = chr(92) * 2
    metrics_tex = ['% Generated from results/metrics.json; historical notebook results.',
        r'\begin{center}', r'\begin{tabular}{lr}', r'\toprule',
        'Метрика & Значение ' + endrow, r'\midrule']
    for title, key in [('Accuracy','accuracy'),('Balanced accuracy','balanced_accuracy'),('ROC-AUC','roc_auc')]:
        metrics_tex.append(title + ' & ' + value(metrics[key]) + ' ' + endrow)
    metrics_tex += [r'\bottomrule', r'\end{tabular}', r'\end{center}']
    (target / 'baseline_metrics.tex').write_text('\n'.join(metrics_tex) + '\n', encoding='utf-8')
    matrix_tex = ['% Generated from results/metrics.json; row/column order HC, ULD.',
        r'\begin{center}',r'\begin{tabular}{lrr}',r'\toprule',
        'Истинный класс & Предсказан HC & Предсказан ULD ' + endrow,r'\midrule']
    for title, row in zip(['HC','ULD'], matrix):
        matrix_tex.append('%s & %d & %d %s' % (title,row[0],row[1],endrow))
    matrix_tex += [r'\bottomrule',r'\end{tabular}',r'\end{center}']
    (target / 'baseline_confusion.tex').write_text('\n'.join(matrix_tex) + '\n', encoding='utf-8')
    print('Generated TeX tables from saved historical metrics; no experiment was run.')



def make_bibliography(root):
    """Render the brace-delimited fields used by this project's Bib file."""
    import re
    source = (root / 'article/references.bib').read_text(encoding='utf-8')
    entries = []
    for start in re.finditer(r'@\w+\s*\{\s*([^,]+),', source):
        pos = start.end()
        fields = {}
        while pos < len(source):
            match = re.match(r'\s*,?\s*(\w+)\s*=\s*\{', source[pos:])
            if not match:
                break
            name = match.group(1)
            pos += match.end()
            opening, depth = pos, 1
            while pos < len(source) and depth:
                char = source[pos]
                if char == chr(92):
                    pos += 2
                    continue
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                pos += 1
            if depth:
                raise ValueError('Unbalanced Bib field')
            fields[name] = source[opening:pos-1]
        if 'title' not in fields:
            raise ValueError('Missing title for ' + start.group(1))
        entries.append((start.group(1),fields))
    if len(entries) != 30:
        raise ValueError('Expected the 30 current bibliography entries')
    accents = {'ë':r'\"{e}', 'í':r"\'{i}", 'ö':r'\"{o}', 'ü':r'\"{u}',
               'ć':r"\'{c}", 'ğ':r'\u{g}'}
    def latex(value):
        for char, command in accents.items():
            value = value.replace(char,command)
        return value
    lines = ['% Generated from article/references.bib; no Biber required.',
        r'\begin{thebibliography}{99}', r'\small']
    for key,fields in entries:
        people = fields.get('author','').split(' and ')
        people = [p for p in people if p and p != 'others']
        authors = '; '.join(people[:3])
        if len(people)>3 or 'others' in fields.get('author',''):
            authors += ' et al.'
        title = fields['title']
        venue = fields.get('journaltitle',fields.get('booktitle',fields.get('publisher','')))
        parts = [authors, r'\textit{'+title+'}', venue, fields.get('year','')]
        text = '. '.join(latex(p.strip().rstrip('.')) for p in parts if p.strip())+'.'
        if fields.get('doi'):
            text += ' DOI: '+r'\url{'+fields['doi']+'}.'
        elif fields.get('url'):
            text += ' '+r'\url{'+fields['url']+'}.'
        lines += [r'\bibitem{'+key+'}',r'\begin{otherlanguage}{english}',text,
                  r'\end{otherlanguage}']
    lines += [r'\end{thebibliography}']
    (root / 'article/generated/bibliography.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Bibliography generated: 30 entries; Biber is not needed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile', action='store_true', help='Also run pdfLaTeX twice if installed.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    make_tables(root)
    make_bibliography(root)
    if args.compile:
        compiler = shutil.which('pdflatex')
        if not compiler:
            parser.exit(1, 'Tables and bibliography generated. PDF needs pdfLaTeX; use Overleaf.\n')
        for _ in range(2):
            subprocess.run([compiler,'-interaction=nonstopmode','-halt-on-error',
                'article/article.tex'],cwd=root,check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
