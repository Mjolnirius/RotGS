import unittest

from tools.reports.ledger_tables import parse_timeline_table


class TimelineTableTests(unittest.TestCase):
    def setUp(self):
        self.header = '| ' + ' | '.join(['ID'] + [f'Column {i}' for i in range(11)]) + ' |'
        self.separator = '| ' + ' | '.join(['---'] * 12) + ' |'

    def row(self, rid):
        return '| ' + ' | '.join([rid] + ['value'] * 11) + ' |'

    def table(self, rows):
        return '\n\n' + self.header + '\n' + self.separator + '\n' + rows + '\n\n'

    def test_latest_runs_belong_to_one_contiguous_table(self):
        ids = [f'R{i:03d}' for i in range(85, 90)]
        headers, rows = parse_timeline_table(self.table('\n'.join(self.row(rid) for rid in ids)))
        self.assertEqual(len(headers), 12)
        self.assertEqual([row[0] for row in rows], ids)

    def test_blank_before_new_run_is_rejected_instead_of_silently_collected(self):
        with self.assertRaisesRegex(ValueError, 'Detached timeline rows'):
            parse_timeline_table(self.table(self.row('R085') + '\n\n' + self.row('R089')))

    def test_text_between_rows_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Detached timeline rows'):
            parse_timeline_table(self.table(self.row('R085') + '\nA note\n' + self.row('R089')))

    def test_wrong_column_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'expected 12'):
            parse_timeline_table(self.table('| R089 | too few cells |'))


if __name__ == '__main__':
    unittest.main()
