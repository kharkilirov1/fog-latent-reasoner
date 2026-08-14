from operator_orbit_scaling_experiment import subgroup


def test_subgroups_have_requested_size_and_are_closed():
    for q in (1,2,3,5,6,10,15,30):
        H=subgroup(q); assert len(H)==q
        S=set(H)
        assert all((a*b)%31 in S for a in H for b in H)
