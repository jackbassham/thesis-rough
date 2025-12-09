    # Shift present day parameters forward one day, for one point Middle Weddell
    ui_t = ui[1:,:,:]
    vi_t = vi[1:,:,:]
    ua_t = ua[1:,:,:]
    va_t = va[1:,:,:]
    ci_t = ci[1:,:,:]
    t_t = time[1:]
    r_t = r[1:,:,:]

    # Remove last day from previous day parameters
    ci_t1 = ci[:-1,:,:]