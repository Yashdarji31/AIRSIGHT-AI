\# AIRSIGHT AI — Data Quality Audit



\*\*Audit date:\*\* 13 November 2026

\*\*Data source:\*\* OpenAQ API v3

\*\*Coverage:\*\* 8 Indian cities



\## 1. Executive Summary



This audit investigates missing observations, sensor coverage, historical prediction eligibility, and data reliability.



Production requirements remain unchanged:

\- Minimum coverage: 75%

\- Minimum history: 24 hours

\- Maximum measurement age: 2 hours



\## 2. Dataset Overview



\- Cities: 8

\- Selected sensor-pollutant records: 48

\- Cleaned historical records: 6,996



These selected sensors do not represent complete city-wide monitoring networks.



\## 3. Historical Eligibility Analysis



| Eligibility | Records | Percentage |

|---|---:|---:|

| 24/24 hours | 80 | 5.24% |

| 23/24 hours | 80 | 5.24% |

| 22/24 hours | 945 | 61.93% |



\- Total candidate records: 1,526

\- Newly admitted at 22/24: 865



These are retrospective results using previously examined test data, not independent validation results.



\## 4. Low-Coverage Findings



\- 1,730 low-coverage appearances

\- 128 unique low-coverage sensor-hours

\- 35 affected sensors

\- Newly admitted low-coverage hours had 25% or 50% coverage



Patterns appeared across all eight cities, particularly around 17:30 and 18:30 UTC.



The underlying cause remains unconfirmed.



\## 5. Raw API Investigation



\### Ahmedabad — CO sensor 12235726



12 raw 15-minute records returned for the four-hour window. No records from 18:00 to 18:45 UTC on 8 November 2026.



\### Ahmedabad — NO2 sensor 12235728



The same 12-record pattern and missing interval were observed.



\### Delhi — PM2.5 sensor 23534



12 hourly records returned for 8 November 2026, with several missing timestamps. Endpoint filtering behavior requires caution.



\### Delhi — PM10 sensor 12234786



\- 54 unique timestamps returned

\- 96 expected 15-minute slots

\- 42 missing slots

\- 56.25% timestamp completeness



Two requests returned identical timestamp patterns. This is timestamp completeness, not official OpenAQ coverage.



\## 6. Conclusions



1\. Missing measurements exist in raw API responses.

2\. Missing data is not solely a CSV-cleaning artifact.

3\. Similar missing intervals appeared on two Ahmedabad sensors.

4\. Delhi sensors also showed incomplete historical availability.

5\. The upstream cause has not been established.



\## 7. Production Decision



Keep the existing configuration:



```python

MIN\_COVERAGE\_PERCENT = 75.0

MIN\_HISTORY\_HOURS = 24

MAX\_DATA\_AGE\_HOURS = 2

```



Do not approve the 22/24-hour alternative for production without independent validation.



\## 8. Recommendations



\- Monitor missing observations by city and pollutant.

\- Preserve raw API responses for diagnostics.

\- Investigate repeated gaps with the data provider.

\- Retain rejected prediction records.

\- Use independent chronological validation for future changes.

\- Clearly distinguish historical demonstrations from live forecasts.



\## 9. Limitations



\- Selected sensors do not provide complete city-wide coverage.

\- The cause of missing observations is unknown.

\- Historical backtest results are not independent guarantees.

\- AIRSIGHT AI does not currently implement a validated AQI calculation.

\- Historical prediction examples are not live forecasts.



\## Final Conclusion



Maintaining strict data-quality requirements is currently the more defensible engineering decision.



Future improvements should prioritize source-data reliability, independent model validation, and transparent reporting.
