@echo off
REM Parcourir tous les fichiers .tex dans le dossier et compiler avec latexmk
for %%f in (*.tex) do (
    latexmk -pdf -silent %%f
)
REM Nettoyer les fichiers auxiliaires (optionnel)
latexmk -c
pause
