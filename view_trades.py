#!/usr/bin/env python3
"""
CLI tool to view trade history
Usage:
    python view_trades.py            # Recent 20 trades
    python view_trades.py --all      # All trades
    python view_trades.py --stats    # Statistics
    python view_trades.py --open     # Current open trade
"""
import argparse
from datetime import datetime
from trade_history import TradeHistory
from tabulate import tabulate


def format_trade(trade):
    """Format trade for display"""
    entry_time = datetime.fromisoformat(trade['entry_time']).strftime('%Y-%m-%d %H:%M')
    exit_time = datetime.fromisoformat(trade['exit_time']).strftime('%Y-%m-%d %H:%M') if trade['exit_time'] else '-'

    return {
        'ID': trade['id'],
        'Time': entry_time,
        'Dir': trade['direction'],
        'Entry': f"{trade['entry_price']:.1f}",
        'Exit': f"{trade['exit_price']:.1f}" if trade['exit_price'] else '-',
        'Exit Type': trade['exit_type'] or '-',
        'PnL': f"{trade['pnl']:+.2f}" if trade['pnl'] is not None else '-',
        'PnL%': f"{trade['pnl_percentage']:+.2f}%" if trade['pnl_percentage'] is not None else '-',
        'Status': trade['status'].upper(),
        'Mode': trade['mode']
    }


def main():
    parser = argparse.ArgumentParser(description='View SimilanCao Trader history')
    parser.add_argument('--all', action='store_true', help='Show all trades')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--open', action='store_true', help='Show open trade')
    parser.add_argument('--limit', type=int, default=20, help='Number of recent trades to show')
    args = parser.parse_args()

    history = TradeHistory()

    if args.stats:
        # Show statistics
        stats = history.get_statistics()
        print("\n" + "=" * 60)
        print("📊 TRADING STATISTICS")
        print("=" * 60)
        print(f"Total Trades:     {stats['total_trades']}")
        print(f"Winning Trades:   {stats['winning_trades']} ({stats['win_rate']:.1f}%)")
        print(f"Losing Trades:    {stats['losing_trades']}")
        print(f"Total PnL:        {stats['total_pnl']:+.2f} USDT")
        print(f"Average PnL:      {stats['avg_pnl']:+.2f} USDT")
        print(f"Best Trade:       {stats['best_trade']:+.2f} USDT")
        print(f"Worst Trade:      {stats['worst_trade']:+.2f} USDT")
        print("=" * 60 + "\n")

    elif args.open:
        # Show open trade
        trade = history.get_open_trade()
        if trade:
            print("\n" + "=" * 60)
            print("📍 CURRENT OPEN TRADE")
            print("=" * 60)
            formatted = format_trade(trade)
            for key, value in formatted.items():
                print(f"{key:15} {value}")
            print(f"{'Reason':15} {trade['reason']}")
            print(f"{'Confidence':15} {trade['confidence']:.0%}" if trade['confidence'] else "")
            print("=" * 60 + "\n")
        else:
            print("\n✓ No open trade\n")

    else:
        # Show recent trades
        limit = None if args.all else args.limit
        trades = history.get_recent_trades(limit=limit or 1000)

        if not trades:
            print("\n✓ No trades found\n")
            return

        print(f"\n📜 {'ALL TRADES' if args.all else f'RECENT {len(trades)} TRADES'}")
        print("=" * 120)

        formatted_trades = [format_trade(t) for t in trades]
        print(tabulate(formatted_trades, headers='keys', tablefmt='grid'))
        print()


if __name__ == '__main__':
    main()
