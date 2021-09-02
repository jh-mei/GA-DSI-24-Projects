# Project 1: Standardized Test Analysis

### Overview

The SAT and ACT are standardized tests that many colleges and universities in the United States require for their admissions process. This score is used along with other materials such as grade point average (GPA) and essay responses to determine whether or not a potential student will be accepted to the university.

The SAT has two sections of the test: Evidence-Based Reading and Writing and Math ([*source*](https://www.princetonreview.com/college/sat-sections)). The ACT has 4 sections: English, Mathematics, Reading, and Science, with an additional optional writing section ([*source*](https://www.act.org/content/act/en/products-and-services/the-act/scores/understanding-your-scores.html)).
* [SAT](https://collegereadiness.collegeboard.org/sat)
* [ACT](https://www.act.org/content/act/en.html)

Standardized tests have long been a controversial topic for students, administrators, and legislators. Since the 1940's, an increasing number of colleges have been using scores from sudents' performances on tests like the SAT and the ACT as a measure for college readiness and aptitude ([*source*](https://www.minotdailynews.com/news/local-news/2017/04/a-brief-history-of-the-sat-and-act/)). Supporters of these tests argue that these scores can be used as an objective measure to determine college admittance. Opponents of these tests claim that these tests are not accurate measures of students potential or ability and serve as an inequitable barrier to entry.

### Problem Statement

SAT and ACT exams have always been a cornerstone of American college culture. It is common knowledge that colleges and universities in the US place great emphasis on SAT or ACT scores as a tool to quantify the readiness of a student for their rigorous curriculum. In fact, some states have compulsory testing on either the SAT or ACT. However, many people refute the effectiveness of this compulsion, which is the basis of this project. 

In this project, my goals are to:

- Determine if compulsory testing means getting good results
- If not, find the underlying reason as to why this is the case
- Find a way to improve the overall results

### Approach

#### Step 1: Cleaning the Data

The first thing I did was to clean the data. I was provided four separate datasets for the ACT and SAT results by state for 2017 and 2018 respectively. Going through the data, I needed to ensure that there were no error or missing values. I also had to check the dtypes of the dataset columns to ensure that everything was as intended. I also merged the four separate datasets into one complete dataset which would make comparing its attributes easier.

This was to prepare the data for the eventual analysis and comparisons which would help to answer my problem statement.


#### Step 2: Exploratory Data Analysis

Next, I went to take a look at the summary statistics of the merged dataset. When comparing the tests by year, I noticed that the 3rd interquartile range was substantially higher than that of the 2nd of the SAT. Furthermore, the median was slightly less than the mean. The same observation could be made for ACT scores, but the implication that can be drawn from this is that the increase in difficulty for the SAT test from 2017 to 2018 was higher than that for the ACT. This might imply that the SAT test is on average harder than the ACT test and thus will be less popular, but this implication is inconclusive.

Following that, I sorted the SAT and ACT scores in descending order and looked at the top and bottom 5. I immediately noticed a large number of states with 100% participation rate. Also, such states which took the ACT with 100% participation rate far outstripped the number of states which took the SAT with 100% participation rate. The fact that any state would have 100% participation rate in any test was suspicious to me, so I confirmed that in the US, graduates from certain states are obliged to take certain tests. Also, I found that fee waivers were a major factor in increases in participation rate.

After that, I went to numerically compare the participation rate and mean results for each respective test. I found out that for states with a low participation rate in a certain test, they would have a relatively high mean result. This led me to suspect that there was a strong negative correlation between participation rate and result.


#### Step 3: Visualization

I first created a heatmap to compare the attributes of the merged dataset. From that, I found several things:

* Students' scores for a given section are almost perfectly correlated (r > 0.97) with other sections of the same test in the same year, which meant that if a student performs well for one section, it is very likely that they would also perform well for other sections.    

* Scores for a given section are less correlated, but still significantly correlated (r > 0.8) with other sections of the same test in different years. In other words, the tests were not of the same difficulty, and the tests in 2018 was harder than that of 2017.

* ACT participation and SAT participation in the same year have a high negative correlation (r < -0.8), which meant that a significant portion of the graduating cohort takes either the ACT or SAT, with few opting to take both.
    
* Test participation for different years have a high correlation (r > 0.87), which implied that most states choose to stick with the same test as the year before.
    
* Test mean composite / total scores are highly negatively correlated (r < -0.84) to their respective participation rates, which meant that the students who take a non-compulsory test in their state are more likely to do well, which corroborates with conclusion 2a.
    
Next, I plotted histograms for the mean total and composite scores for the tests and found that the distribution was bimodal. The two peaks implied that there were two categories of students. The category with a lower average score but higher count peak was likely the students in states with 100% participation rate. The category with a higher average score but with lower count peak was likely the students who opted to take tests in non-compulsory states and performed well.

Next, I plotted a box and scatter diagram which showed that majority of students taking either test were in states that had compulsory testing.

Finally, I plotted scatterplots to corroborate the conclusions made from the heatmap as well as check for any gross outliers, to which there were actually a handful, but they were not significant enough to alter the correlation.


### Conclusion and Statement


Based on the exploration of the data, we can come to the conclusion that mandating tests does not fare well for the average student. If a student wishes to take either the SAT or ACT, they would naturally work hard for it. There would not be a necessity for compulsion.

Thus, it is advisable to make taking the tests non-compulsory.

Furthermore, we can also see that waiving fees for the tests (in non-compulsory states) increases the participation rate of the state, but also reduces the mean total / composite scores. This implies that there was an increase in students who took the test for no reason other than because it was free.

Thus, it is advisable to remove the waiver of fees as well.

Furthermore, we can also come to the conclusion that it is worth it to take a closer look at states with a low participation rate in either test and also low mean total / composite scores as that might reflect a low standard of education in those states.

Thus, it is advisable to reinvest the fee waiver into trying to improve the standard of education in those states instead.