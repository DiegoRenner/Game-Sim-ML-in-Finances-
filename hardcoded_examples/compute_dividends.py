
# takes dividend paid per stock and number of stocks as well as price of individual stock
# returns total dividend payout
def comp_div(d, D, stocks):
    dividend = d*D*stocks
    return dividend