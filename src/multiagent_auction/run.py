from multiagent_auction.argparser import load_args
from multiagent_auction.experiment import AuctionSimulationRunner

def main() -> None:
    """
    Entry point for running the auction simulation.
    Retrieves parameters, initializes the simulation runner and executes the simulation.
    """
    args = load_args()
    runner = AuctionSimulationRunner(*args)
    runner.execute()

if __name__ == '__main__':
    main()



"""
Próximas implementações:

1. Eficiência de alocação
- Métrica: % de episódios em que o bem foi para o maior valor.
- Log: winner_idx, values_log (p/ comparar com argmax).
- Visual: número único + linha (média móvel) ao longo dos episódios.

2. Distribuição de vencedores (por jogador)
- Métrica: fração de vitórias por agente (win rate).
- Log: winner_idx, N.
- Visual: barras (Player 1..N) com rótulo do valor em cada barra.

3. Custo dos perdedores (partial/all-pay)
- Métrica: média e p95 do pagamento dos não-vencedores por episódio.
- Log: payments_log (por agente/episódio), winner_idx.
- Visual: dois números (média, p95) + linha de média móvel.

4. Estabilidade
- Métrica: desvio-padrão dos lances por episódio (ou variação entre episódios).
- Log: bids_log completo.
- Visual: linha (média móvel do desvio-padrão) + boxplot da última janela.

5. Resumo por estágios de t (smooth TL)
- Métricas por estágio t_k: eficiência, custo dos perdedores (média), erro teórico médio.
- Log: t_stage_idx (ou t por episódio) para agregação por estágio.
- Visual:
--> Tabela com linhas para cada t_k e colunas: Eficiência | Custo perd. (média) | Erro médio.
--> Opcional: gráfico com t no eixo x e as 3 curvas em linhas separadas.
"""