# from unnamed_rcv_thing.loaders import load_csv
# from unnamed_rcv_thing.cleaning import clean_rule1, clean_rule2
# from unnamed_rcv_thing.elections import run_election, RCVStep
# from unnamed_rcv_thing.visualization import plot_results


# def test_e2e_simple():
#     """simple example of what a "full" use would look like"""

#     # load CVR -> PP representation
#     pp = load_csv("example.cvr")

#     # apply rules to get new PP
#     cleaned_pp = clean_rule2(clean_rule1(pp))

#     # write intermediate output for inspection
#     cleaned_pp.save("cleaned.cvr")

#     # run election using a configured RCV step object
#     outcome = run_election(cleaned_pp, RCVStep())

#     plot_results(outcome)
