# Resultados finales (TweetToxic)

F1 con umbral de meseta elegido sin mirar el fold de validación y jerarquía HS=0 ⇒ TR=0 (notebook 07). 'F1 macro (registro)': protocolo de los notebooks 04–06 (umbral argmax de los otros folds). CV (s) y tamaño de un ensamble: suma de sus modelos base.

| familia | modelo | F1 HS | F1 TR | F1 AG | F1 macro | desv. | F1 micro | F1 macro folds_seed2 | desv. folds_seed2 | F1 micro folds_seed2 | F1 macro (registro) | Kaggle público | CV (s) | tamaño (MB) | envío final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lineal | V2__logreg__nested | 0.781 | 0.832 | 0.758 | 0.7902 | 0.0192 | 0.7859 | 0.7852 | 0.0191 | 0.7811 | 0.7937 | 0.77714 | 48.0 | 0.52 | — |
| bayes | V4__cnb__word | 0.743 | 0.784 | 0.703 | 0.7437 | 0.0281 | 0.741 | 0.7435 | 0.0211 | 0.7414 | 0.7461 | — | 0.2 | 2.38 | — |
| kernel | V3__rbfsvm__nested | 0.787 | 0.812 | 0.746 | 0.7815 | 0.0229 | 0.7792 | 0.7856 | 0.0237 | 0.7834 | 0.7845 | — | 104.2 | 18.19 | — |
| árboles | V3__et__word__k2000 | 0.76 | 0.81 | 0.72 | 0.7631 | 0.0206 | 0.7594 | 0.7519 | 0.0129 | 0.749 | 0.767 | — | 2.0 | 35.25 | — |
| boosting | V1__lgbm__word+hand__k2000 | 0.743 | 0.805 | 0.725 | 0.7575 | 0.0091 | 0.7528 | 0.7558 | 0.0245 | 0.752 | 0.7571 | — | 1.5 | 1.5 | — |
| MLP | V3__nn_mlp__tune-d0.5-lr0.002 | 0.758 | 0.773 | 0.724 | 0.7515 | 0.0099 | 0.75 | 0.7554 | 0.0169 | 0.7531 | 0.755 | — | 21.2 | 67.26 | — |
| CNN | V1__nn_cnn_char__tune-d0.3-lr0.002 | 0.761 | 0.793 | 0.732 | 0.7621 | 0.0232 | 0.7591 | 0.767 | 0.0205 | 0.7621 | 0.7654 | — | 347.6 | 0.42 | — |
| BiGRU/fastText | V3__nn_bigru__w2v | 0.713 | 0.785 | 0.691 | 0.7298 | 0.0101 | 0.7245 | 0.734 | 0.0244 | 0.7297 | 0.733 | — | 46.6 | 3.27 | — |
| Transformer | V1__nn_trf_bpe__v800-d128-l2-aug | 0.699 | 0.735 | 0.675 | 0.7029 | 0.0237 | 0.6997 | 0.694 | 0.0258 | 0.6909 | 0.7052 | — | 296.6 | 5.85 | — |
| ensamble | D · (iv) stacking (todas las etiquetas) (candidato) | 0.789 | 0.84 | 0.774 | 0.8008 | 0.018 | 0.7968 | 0.7961 | 0.0212 | 0.7927 | — | — | 606.1 | 112.72 | FINAL_A |
| final | V1__logreg__nested | 0.777 | 0.826 | 0.759 | 0.7871 | 0.0166 | 0.7832 | 0.793 | 0.0157 | 0.7893 | 0.7877 | — | 46.8 | 0.42 | FINAL_B |
