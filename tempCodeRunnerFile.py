# Plotar a curva ROC para cada classe
plt.figure()
colors = ['blue', 'red', 'green', 'purple', 'orange']
for i in range(len(genre_keywords)):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'Curva ROC para {genre_keywords[i]} (área = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC para cada classe')
plt.legend(loc='lower right')
plt.show()