So the driver would do something like:                                                                      03:01 [15/1082]

  1. approximate x from its CF using a convergent deep enough for the current target precision
  2. evaluate sin(approx_x) at dps
  3. extract CF terms from that result
  4. repeat at a higher dps
  5. only keep terms that match across runs

  The important part is that you do not trust a term just because one precision level produced it. You trust it only when the
  next higher precision level produces the same term too.
