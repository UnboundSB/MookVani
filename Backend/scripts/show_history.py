import json, os

h = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "models", "phase4_word_model", "history.json")))
eps = len(h['tl'])
va_list = h['va']
best_ep = va_list.index(max(va_list)) + 1
print(f"Epochs: {eps}")
print(f"Best Val Acc : {max(va_list)*100:.2f}%  (Epoch {best_ep})")
print(f"Final Val Acc: {va_list[-1]*100:.2f}%")
print(f"Best Val F1  : {max(h['vf1']):.4f}")
print()
print(f"{'Ep':>3}  {'Phase':>7}  {'TrainAcc':>9}  {'ValAcc':>8}  {'F1':>7}  {'LR_rnn':>10}  {'LR_trans':>10}")
print("-" * 72)
for i in range(eps):
    phase = "FROZEN" if (i+1) <= 30 else "FULL"
    marker = " <--best" if (i+1) == best_ep else ""
    print(f"{i+1:>3}  {phase:>7}  {h['ta'][i]*100:>8.1f}%  "
          f"{h['va'][i]*100:>7.1f}%  {h['vf1'][i]:>7.4f}  "
          f"{h['lr_rnn'][i]:>10.2e}  {h['lr_trans'][i]:>10.2e}{marker}")
