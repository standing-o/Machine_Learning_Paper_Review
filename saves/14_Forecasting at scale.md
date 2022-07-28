## Forecasting at scale
- Authors : Taylor, Sean J and Letham, Benjamin
- Journal : The American Statistician
- Year : 2018
- Link : https://peerj.com/preprints/3190.pdf

###  Abstract
- We propose a modular regression model with interpretable parameters that can be intuitively adjusted by analysts with domain knowledge about the time series.

### **Introduction**
- Main themes in the practice of creating business forecasts:
1. Completely automatic forecasting techniques can be hard to tune and are often too inflexible to incorporate useful assumptions or heuristics.
2. The analysts responsible for data science tasks throughout an organization typically have deep domain expertise about the specific products or services that they support, but often do not have training in time series forecasting.
- We intend to provide some useful guidance for producing forecasts at **scale**.
  1.  Business forecasting methods should be suitable for a large number of people making forecasts, possibly without training in time series methods.
  2.  a large variety of forecasting problems with potentially idiosyncratic features. 
  3.  In most realistic settings, a large number of forecasts will be created, necessitating efficient, automated means of evaluating and comparing them, as well as detecting when they are likely to be performing poorly.

- Schematic view of the analyst-in-the-loop approach to forecasting at scale, which best makes use of human and automated tasks.
  <img src="https://user-images.githubusercontent.com/57218700/177343533-32bc85c3-289e-4bd5-ad40-437806ff02fc.png" width=30%>

- The number of events created on Facebook.
  <img src="https://user-images.githubusercontent.com/57218700/177343612-ad68206a-7cb9-4121-bd94-f44258c71645.png" width=70%>

### **Features of business time series**
- There are some features common to many of business forecasting problems. 
- There are several seasonal effects clearly visible in this time series: weekly and yearly cycles, and a pronounced dip around Christmas and New Year.
- The time series also shows a clear change in trend in the last six months, which can arise in time series impacted by new products or market changes.
- Real datasets often have outliers and this time series is no exception.
- (baseline) Forecasts on the time series from previous figure using `forecast` package in R.
  <img src="https://user-images.githubusercontent.com/57218700/177345320-f653a9f4-d2cd-46fb-9407-ec7987a9aff7.png" width=70%>
1. `auto.arima` : ARIMA model
➔ Large trend errors when there is a change in trend near the cutoff period and they fail to capture any seasonality.
2. `ets` : Exponential smoothing model
➔  Capture weekly seasonality but miss longer-term seasonality
3. `snaive` :  a random walk model that makes constant predictions with weekly seasonality (seasonal naive)
➔  Capture weekly seasonality but miss longer-term seasonality
4. `tbats` : TBATS model

➔ All of the methods overreact to the end-of-year dip because they do not adequately model yearly seasonality.

### **The `Prophet` Forecasting Model**
- Time series forecasting model designed to handle the common features of business time series.
- It is also designed to have intuitive parameters that can be adjusted without knowing the details of the underlying model.
- We use a decomposable time series model (Harvey & Peters 1990) with three main model components: trend, seasonality, and holidays:
  <img src="https://user-images.githubusercontent.com/57218700/177347057-64add42c-1b25-420f-8542-fd4c3b10f0ff.png" width=30%>
  - g(t) : the trend function which models non-periodic changes in the value of the time series
  - s(t) : periodic changes
  - h(t) : the effects of holidays which occur on potentially irregular schedules over one or more days
  - &epsilon;<sub>t</sub> : any idiosyncratic changes which are not accommodated by the model (normally distributed)
- This specification is similar to a generalized additive model (GAM) (Hastie & Tibshirani 1987), a class of regression models with potentially non-linear smoothers applied to the regressors. 
  ➔ We use only time as a regressor but possibly several linear and non-linear functions of time as components.
  ➔ It decomposes easily and accommodates new components as necessary, for instance when a new source of seasonality is identified.
  ➔ The user can interactively change the model parameters.
  
- We are framing the forecasting problem as a curve-fitting exercise, which is inherently different from time series models that explicitly account for the temporal dependence structure in the data.
- Practical advantages:
1. Flexibility: We can easily accommodate seasonality with multiple periods and let the analyst make different assumptions about trends.
2. Unlike with ARIMA models, the measurements do not need to be regularly spaced, and we do not need to interpolate missing values e.g. from removing outliers.
3. Fitting is very fast
4. Easily interpretable parameters that can be changed by the analyst to impose assumptions on the forecast

#### The trend model (saturating growth model and a piece wise linear model)
- **Nonlinear, Saturating Growth**
  ➔ For growth forecasting, the core component of the data generating process is a model for how the population has grown and how it is expected to continue growing.
  ➔ This sort of growth is typically modeled using the logistic growth model, which in its most basic form is:
  <img src="https://user-images.githubusercontent.com/57218700/177349082-2f8b4a35-53ea-4f29-b7ee-4d2d2926df76.png" width=30%>
  with C the carrying capacity, k the growth rate, and m an offset parameter.
  - The carrying capacity is not constant. We thus replace the fixed capacity C with a time-varying capacity C(t)
  - The growth rate is not constant.
  ➔ We incorporate trend changes in the growth model by explicitly defining changepoints where the growth rate is allowed to change.
  - Define a vector of rate adjustments &delta; ∈ R<sup>s</sup>, where &delta;<sub>j</sub> is the change in rate that occurs at time s<sub>j</sub>. k is the base rate and a<sub>j</sub>(t) = 1 (if t≥s<sub>j</sub>) or 0 (otherwise).
  - The rate at time t is then k + a(t)<sup>T</sup>&delta;. 
  - When the rate k is adjusted, the offset parameter m must also be adjusted to connect the endpoints of the segments. The correct adjustment at changepoint j is easily computed as:
    <img src="https://user-images.githubusercontent.com/57218700/177350453-ec18288a-44c4-447c-88fe-bdafbaaccb10.png" width=45%>
  - The piecewise logistic growth model:
    <img src="https://user-images.githubusercontent.com/57218700/177350495-472b0b29-ac33-4e94-8e78-febb94473146.png" width=45%>
  
- **Linear Trend with Changepoints**
  ➔ For forecasting problems that do not exhibit saturating growth, a piece-wise constant rate of growth provides a parsimonious and often useful model.
  <img src="https://user-images.githubusercontent.com/57218700/177350851-f0a89289-4842-4911-b5b2-8fe8bee43cd9.png" width=45%>
  
- **Automatic Changepoint Selection**
  ➔ The changepoints s<sub>j</sub> could be specified by the analyst using known dates of product launches and other growth-altering events, or may be automatically selected given a set of candidates.
  ➔ We often specify a large number of changepoints and use the prior δ<sub>j</sub> ∼ Laplace(0, τ ).
  ➔ A sparse prior on the adjustments δ has no impact on the primary growth rate k, so as τ goes to 0 the fit reduces to standard
  (not-piecewise) logistic or linear growth.
  
- **Trend Forecast Uncertainty**
  ➔ Each of which has a rate change δ<sub>j</sub> ∼ Laplace(0, τ). We simulate future rate changes that emulate those of the past by replacing τ with a variance inferred from data.
  ➔ The maximum likelihood estimate of the rate scale parameter: &lambda; = mean(abs(&delta;<sub>j</sub>))
  ➔ The average frequency of changepoints matches that in the history:
  <img src="https://user-images.githubusercontent.com/57218700/177351618-3b4ddefb-d4b7-42e8-9f6b-874c2af5f4d7.png" width=45%>

#### Seasonality
- We must specify seasonality models that are periodic functions of t.
- We rely on Fourier series to provide a flexible model of periodic effects. We can approximate arbitrary smooth seasonal effects:
  <img src="https://user-images.githubusercontent.com/57218700/177352190-6c1966be-267a-48a3-a32f-cff624c00946.png" width=50%>
-  Fitting seasonality requires estimating the 2N parameters &beta; = [a<sub>1</sub>, b<sub>1</sub>, . . . , a<sub>N</sub> , b<sub>N</sub>]<sup>T</sup>.
- The seasonal component is s(t) = X(t)&beta;.
- In our generative model we take &beta; ∼ Normal(0, σ<sup>2</sup>) to impose a smoothing prior on the seasonality.
- For yearly and weekly seasonality we have found N = 10 and N = 3 respectively to work well for most problems.

#### Holidays and Events
- Holidays and events provide large, somewhat predictable shocks to many business time series and often do not follow a periodic pattern, so their effects are not well modeled by a smooth cycle.
- The impact of a particular holiday on the time series is often similar year after year, so it is important to incorporate it into the forecast.
- Incorporating this list of holidays into the model is made straightforward by assuming that the effects of holidays are independent.
- For each holiday i, let D<sub>i</sub> be the set of past and future dates for that holiday. 
- We add an indicator function representing whether time t is during holiday i, and assign each holiday a parameter κi which is the corresponding change in the forecast.
  <img src="https://user-images.githubusercontent.com/57218700/177353342-44f85b3f-5975-4904-a123-a2f58a66f3d4.png" width=35%>
  ➔ It is often important to include effects for a window of days around a particular holiday.
  ➔ To account for that we include additional parameters for the days surrounding the holiday.

#### Model Fitting
- `Prophet` model forecasts made on the same three dates as baseline:
  <img src="https://user-images.githubusercontent.com/57218700/177353716-b74d4c69-41f7-4f19-81e4-4f95c583e7ec.png" width=70%>
  ➔ The `Prophet` forecast is able to predict both the weekly and yearly seasonalities, and unlike the baselines, does not overreact to the holiday dip in the first year.
- Forecast incorporating the most recent three months of data exhibits the trend change:
  <img src="https://user-images.githubusercontent.com/57218700/177353773-c9450603-755c-448e-837d-fef12cdb1853.png" width=70%>

- The trend, weekly seasonality, and yearly seasonality components corresponding to the last forecast:
  <img src="https://user-images.githubusercontent.com/57218700/177354176-e13d2330-3403-4671-a0b8-1ccdadfcbbff.png" width=60%>
  ➔ An important benefit of the decomposable model is that it allows us to look at each component of the forecast separately.
- The parameters &tau; and &sigma; are controls for the amount of regularization on the model changepoints and seasonality respectively. (to avoid overfitting)


#### Analyst-in-the-Loop Modeling
- In the `Prophet` model specification there are several places where analysts can alter the model to apply their expertise and external knowledge:
  - Capacities, Changepoints, Holidays and seasonality
  - Smoothing parameters : The τ parameter can be turned to increase or decrease the trend flexibility, and σ to increase or decrease the strength of the seasonality component.

### **Automating Evaluation of Forecasts**
- We outline a procedure for automating forecast performance evaluation, by comparing various methods and identifying forecasts where manual intervention may be warranted.

#### Use of Baseline Forecasts
- We prefer using simplistic forecasts that make strong assumptions about the underlying process but that can produce a reasonable forecast in practice.

####  Modeling Forecast Accuracy
- For any forecast with daily observations, we produce up to H estimates of future states that will each be associated with some error. We need to declare a forecasting objective to compare methods and track performance.

####  Simulated Historical Forecasts
- It is difficult to use a method like cross validation because the observations are not exchangeable.
- We use simulated historical forecasts (SHFs) by producing K forecasts at various cutoff points in the history, chosen such that the horizons lie within the history and the total error can be evaluated.
- SHFs simulate the errors we would have made had we used this forecasting method at those points in the past. 
- Issues to be aware of when using the SHF methodology to evaluate and compare forecasting approaches:
  - The more simulated forecasts we make, the more correlated their estimates of error are.
  - Forecasting methods can perform better or worse with more data. Smoothed mean absolute percentage errors for the forecasting methods:
  <img src="https://user-images.githubusercontent.com/57218700/177355867-c283f1a4-b325-4cd6-9e31-b0d375aeffa0.png" width=45%>
  ➔ `Prophet` has lower prediction error across all forecast horizons.

#### Identifying Large Forecast Errors
- It is important to be able to automatically identify forecasts that may be problematic.
- There are several ways that SHFs can be used to identify likely problems with the forecasts:
  - When the forecast has large errors relative to the baselines, the model may be misspecified. 
  ➔ Adjust the trend model or the seasonality
  - Large errors for all methods on a particular date are suggestive of outliers.
  ➔ Remove outliers
  - When the SHF error for a method increases sharply from one cutoff to the next, it could indicate that the data generating process has changed.
  ➔ Adding changepoints or modeling different phases separately may address the issue.

### **Conclusion**
- Components of our forecasting system:
  - The new model that we have developed over many iterations of forecasting a variety of data at Facebook.
  - A system for measuring and tracking forecast accuracy, and flagging forecasts that should be checked manually to help analysts make incremental improvements.