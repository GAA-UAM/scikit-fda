The functions `root_mean_squared_error` and `root_mean_squared_log_error` were added in scikit-learn 1.4 and the parameter `squared` was deprecated in the functions `mean_squared_error` and `mean_squared_log_error` , and removed in version 1.6.

We need to add a `FData`-aware version of these functions and deprecate that parameter.

Desired functionality
Add `FData`-aware functions `root_mean_squared_error` and `root_mean_squared_log_error` to the `scoring` module.
Deprecate `squared` parameter in functions `mean_squared_error`and `mean_squared_log_error`.