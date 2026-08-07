from fiber_flagship.utils import delete_positions, insert_hidden_bits


def test_insert_delete_inverse_single():
    base = 0b101101
    for j in range(7):
        for b in (0, 1):
            word = insert_hidden_bits(base, 6, (j,), b)
            assert delete_positions(word, (j,), 7) == base


def test_insert_delete_inverse_two():
    base = 0b10101
    positions = (1, 5)
    for hidden in range(4):
        word = insert_hidden_bits(base, 5, positions, hidden)
        assert delete_positions(word, positions, 7) == base
