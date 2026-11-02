# Plan de répartition du projet `ldapkg` — équipe de 3

> **Rendu** : dimanche 15/11/2026 au soir · **Soutenances** : semaine du 29/11
> **À mettre à la racine du dépôt GitHub** (à côté de `README.md`) pour que toute l'équipe l'ait sous les yeux.

Ce document est la **carte** du projet. Il répond à quatre questions :
1. Qui est responsable de quel fichier et de quelle fonction ?
2. Dans quel ordre travaille-t-on ?
3. Comment fusionner le travail de trois personnes sans tout casser ?
4. Comment sait-on que c'est fini ?

---

## Table des matières

0. [Comment lire ce document](#0-comment-lire-ce-document)
1. [Point de départ : ce qui existe déjà](#1-point-de-départ--ce-qui-existe-déjà)
2. [Les trois rôles en un coup d'œil](#2-les-trois-rôles-en-un-coup-dœil)
3. [Quel rôle choisir ? (conseils)](#3-quel-rôle-choisir--conseils)
4. [Étape 0 — à faire ensemble avant de se séparer](#4-étape-0--à-faire-ensemble-avant-de-se-séparer)
5. [Rôle A — Moteur statistique](#5-rôle-a--moteur-statistique)
6. [Rôle B — Catégorielles, sélection, évaluation](#6-rôle-b--catégorielles-sélection-évaluation)
7. [Rôle C — Interface, documentation, intégration](#7-rôle-c--interface-documentation-intégration)
8. [Les contrats d'interface (à ne pas casser)](#8-les-contrats-dinterface-à-ne-pas-casser)
9. [Git et GitHub pas à pas dans RStudio](#9-git-et-github-pas-à-pas-dans-rstudio)
10. [Règles anti-conflit](#10-règles-anti-conflit)
11. [Calendrier](#11-calendrier)
12. [Intégration : test de fumée et check-list de fusion](#12-intégration--test-de-fumée-et-check-list-de-fusion)
13. [Le rapport LaTeX à trois](#13-le-rapport-latex-à-trois)
14. [Check-list finale de rendu](#14-check-list-finale-de-rendu)
15. [Préparer la soutenance](#15-préparer-la-soutenance)
16. [Si tu avances seul ou si un camarade décroche](#16-si-tu-avances-seul-ou-si-un-camarade-décroche)
17. [Glossaire](#17-glossaire)

---

## 0. Comment lire ce document

| Ta situation | Lis dans cet ordre |
|---|---|
| **Tu travailles seul pour l'instant** | §1 → §2 → §3 → ton rôle (§5, §6 ou §7) → §9 → §16 |
| **L'équipe est réunie** | Réunion de 30 min : §1 à §3 ensemble, choix des rôles, puis §4 ; ensuite chacun lit son rôle et §8 à §10 |
| **Tu as un conflit Git** | §9.6 |
| **Tu veux savoir si tu as fini** | Le bloc « Définition de terminé » de ton rôle, puis §12 et §14 |

Convention : les cases `- [ ]` sont à cocher dans ton éditeur ou à recopier en *Issues* GitHub (voir §9.8).

---

## 1. Point de départ : ce qui existe déjà

Le fichier `ldapkg.zip` contient un **squelette complet** du package : classe R6, calculs, sélection de variables, appli Shiny, tests, tutoriel, README.

> ⚠️ **Ce squelette n'a jamais été exécuté sous R.** Les formules ont été vérifiées dans un autre langage, mais le code R peut contenir de petites erreurs.

Conséquence importante : **la répartition ne consiste pas à écrire de zéro.** Chaque personne doit, pour sa partie :

1. **Prendre la propriété** des fichiers listés dans son rôle ;
2. **La faire tourner** (`devtools::load_all()`) ;
3. **La corriger** (faire passer ses tests) ;
4. **L'étendre** (cas limites, améliorations) ;
5. **La documenter** (aide en anglais, commentaires) ;
6. **La décrire dans le rapport** (formules, pseudo-code).

### 1.1 Schéma des dépendances

```
 données (X, y)
       │
       ▼
┌──────────────────┐
│ preprocess.R (A) │  encodage : variables catégorielles → colonnes 0/1
└────────┬─────────┘
         ▼
┌──────────────────┐
│ core.R (A)       │  matrices E, B, T · Cholesky · Wilks · axes · coefficients
└────────┬─────────┘
         │
   ┌─────┼──────────────────┬───────────────────┐
   ▼                        ▼                   ▼
selection.R (B)       importance.R (B)     metrics.R (B)
sélection pas à pas   Wilks partiel / F    rappel, précision, F1
   └─────┬──────────────────┴───────────────────┘
         ▼
┌──────────────────────────────────────────┐
│ LDAClassifier.R (classe R6)              │
│  fit / predict / predict_proba  (A)      │
│  cv / class_metrics / summary   (B)      │
│  print / plot / describe_inputs (C)      │
└─────────────────┬────────────────────────┘
                  ▼
   ┌──────────────┼───────────────────┐
   ▼              ▼                   ▼
 app.R (C)   tutorial, README (C)   tests (A et B)
```

Lecture : une flèche signifie « utilise ». Si **A** change le format de ce que renvoie `core.R`, tout ce qui est en dessous peut casser. D'où les **contrats** du §8.

### 1.2 Les exigences du sujet → qui les couvre

| Exigence du cahier des charges | Où dans le code | Rôle |
|---|---|---|
| Cœur de l'algorithme efficace | `core.R` (`.suff_stats`, `.lda_solve`) | **A** |
| Cible catégorielle ≥ 2 modalités | `LDAClassifier.R` (`fit`) | **A** |
| Coefficients du modèle | `.lda_solve` → `$coefficients`, `$intercept` | **A** |
| Évaluation globale (Wilks, F de Rao, axes) | `.lda_solve`, `.rao_F` | **A** |
| Pertinence des variables | `importance.R` (`.partial_table`) | **B** |
| Stratégie pour les catégorielles | `preprocess.R` (codage 0/1) + test par bloc | **A** (code) / **B** (validation, tests, rapport) |
| Stratégie de sélection de variables | `selection.R` | **B** |
| Classe R6 : constructeur, `fit`, `predict`, `predict_proba`, `print`, `summary`, propriétés | `LDAClassifier.R` | **A** / **B** / **C** |
| Application Shiny | `inst/shiny/ldaapp/app.R` | **C** |
| Aide du package en anglais | commentaires roxygen `#'` | **Tous** (chacun pour son code), **C** relit |
| Installable depuis GitHub, jeux de données | `DESCRIPTION`, `inst/extdata/`, `examples.R` | **C** |
| Tutoriel reproductible en anglais | `tutorial/tutorial.Rmd` | **C** |
| Copie `.tar.gz` | `devtools::build()` | **C** |
| Rapport LaTeX (12 pages max) | `report/` | **Tous** (§13) |
| Options bonus (régularisation, grande dimension…) | à décider | **A** / **B** |

---

## 2. Les trois rôles en un coup d'œil

| | **Rôle A — Moteur statistique** | **Rôle B — Catégorielles, sélection, évaluation** | **Rôle C — Interface, documentation, intégration** |
|---|---|---|---|
| **Mission** | Faire calculer le modèle **juste et vite** | Dire **quelles variables comptent**, comment les **choisir**, et si le modèle est **bon** | Rendre le package **installable, utilisable, documenté et présenté** |
| **Fichiers** | `R/preprocess.R` · `R/core.R` | `R/selection.R` · `R/importance.R` *(nouveau)* · `R/metrics.R` *(nouveau)* | `inst/shiny/ldaapp/app.R` · `R/run_app.R` · `R/examples.R` · `R/methods.R` · `R/ldapkg-package.R` · `README.md` · `tutorial/` · `DESCRIPTION` · `NAMESPACE` · `man/` · `.github/` |
| **Méthodes de la classe R6** | `initialize` · `fit` · `predict_proba` · `predict` · `transform` · privées `check_fitted`, `prepare_xy`, `resolve_priors`, `design` | `cv` · `class_metrics` · `summary` | `print` · `plot` · `describe_inputs` |
| **Tests** | `test-core.R` · `test-robustness.R` | `test-categorical.R` · `test-selection.R` · `test-cv.R` | tests des fonctions utilitaires (`ldapkg_example`, `run_app`) + scénarios manuels Shiny |
| **Sections du rapport** | 2, 3, 5a | 4, 5b, 6, 7 | 1, 8, 9, 10, 11 + assemblage |
| **Bonus possible** | Régularisation (`shrinkage`) | DISQUAL | Benchmark grande volumétrie |
| **Difficulté statistique** | ★★★ | ★★ | ★ |
| **Difficulté outils** (R, Git, Shiny) | ★★ | ★★ | ★★★ |
| **Rôle de relecteur** | relit le travail de **C** | relit le travail de **A** | relit le travail de **B** |

La **relecture croisée** (A→C, B→A, C→B) n'est pas une formalité : c'est ce qui force chacun à **comprendre une deuxième partie du projet**, ce qui sera utile en soutenance.

---
