#!/usr/bin/env python
# coding: utf-8

# In[6]:


from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum
from pulp import PULP_CBC_CMD
import pandas as pd
import streamlit as st
from streamlit_calendar import calendar
from datetime import datetime, timedelta
from collections import defaultdict

# Load the datasets
def load_data():
    cover_requirements = pd.read_csv('cover_requirements.csv')
    shift_off_requests = pd.read_csv('shift_off_requests.csv')
    shift_on_requests = pd.read_csv('shift_on_requests.csv')
    staff = pd.read_csv('staff.csv')
    days_off = pd.read_csv('days_off.csv')
    return cover_requirements, shift_off_requests, shift_on_requests, staff, days_off

# Load cover requirements as a dictionary with keys like (day, shift)
def parse_cover_requirements(cover_requirements):
    return {
        (day, shift): requirement 
        for day, shift, requirement in cover_requirements.itertuples(index=False)
    }

# Manually read the days_off CSV file line-by-line to properly parse days off requests
def parse_days_off(days_off):
    days_off_dict = {}
    for _, row in days_off.iterrows():
        employee_id = row['# EmployeeID']
        days_off_str = str(row[' DayIndexes (start at zero)'])  # Convert to string first
        days_off = list(map(int, days_off_str.split(',')))
        days_off_dict[employee_id] = days_off
    return days_off_dict

# Parse max shifts and other constraints from the staff data
def parse_max_shifts(staff):
    max_shifts_dict = {}
    for _, row in staff.iterrows():
        nurse_id = row['# ID']
        max_shifts_info = row[' MaxShifts'].split('|')
        max_shifts = {shift.split('=')[0]: int(shift.split('=')[1]) for shift in max_shifts_info}
        max_shifts_dict[nurse_id] = max_shifts
    return max_shifts_dict

# Create the schedule while including all constraints
def create_schedule(staff, cover_requirements, days_off, shift_on_requests, shift_off_requests):
    nurses = staff['# ID'].tolist()
    days = cover_requirements['# Day'].unique().tolist()
    shift_types = cover_requirements[' ShiftID'].unique().tolist()

    cover_requirements_dict = parse_cover_requirements(cover_requirements)
    days_off_dict = parse_days_off(days_off)
    max_shifts_dict = parse_max_shifts(staff)

    max_total_minutes = staff.set_index('# ID')[' MaxTotalMinutes'].to_dict()
    max_consecutive_shifts = staff.set_index('# ID')[' MaxConsecutiveShifts'].to_dict()

    # Decision Variables
    x = {
        (nurse_id, day_id, shift_id): LpVariable(name=f"x_{nurse_id}_{day_id}_{shift_id}", cat='Binary')
        for nurse_id in nurses
        for day_id in days
        for shift_id in shift_types
    }

    # Create the optimization problem
    model = LpProblem(name="nurse-scheduling", sense=LpMinimize)

    # Objective Function
    shift_on_requests_dict = shift_on_requests.set_index(['# EmployeeID', ' Day', ' ShiftID'])[' Weight'].to_dict()
    shift_off_requests_dict = shift_off_requests.set_index(['# EmployeeID', ' Day', ' ShiftID'])[' Weight'].to_dict()

    # Step 1: Add auxiliary variables for excess nurses assigned beyond the cover requirements
    excess_vars = LpVariable.dicts("excess", [(day_id, shift_id) for day_id in days for shift_id in shift_types], lowBound=0, cat='Continuous')

    # Step 2: Add constraints to define excess variables
    for day_id in days:
        for shift_id in shift_types:
            if (day_id, shift_id) in cover_requirements_dict:
                requirement = cover_requirements_dict[(day_id, shift_id)]
                model += (lpSum(x[(nurse_id, day_id, shift_id)] for nurse_id in nurses) - requirement <= excess_vars[(day_id, shift_id)], f"excess_cover_{day_id}_{shift_id}")
                model += (excess_vars[(day_id, shift_id)] >= 0, f"excess_non_negative_{day_id}_{shift_id}")

    # Labor Law Constraints
    for nurse_id in nurses:
        for day_id in range(len(days) - 1):
            # Rule 1: A nurse cannot work an Early shift (E) if they worked a Day shift (D) the previous day
            model += (x[(nurse_id, day_id, 'D')] + x[(nurse_id, day_id + 1, 'E')] <= 1, f"no_D_to_E_{nurse_id}_{day_id}")
            # Rule 2: A nurse cannot work a Day shift (D) if they worked a Late shift (L) the previous day
            model += (x[(nurse_id, day_id, 'L')] + x[(nurse_id, day_id + 1, 'D')] <= 1, f"no_L_to_D_{nurse_id}_{day_id}")
            # Rule 3: A nurse cannot work an Early shift (E) if they worked a Late shift (L) the previous day
            model += (x[(nurse_id, day_id, 'L')] + x[(nurse_id, day_id + 1, 'E')] <= 1, f"no_L_to_E_{nurse_id}_{day_id}")

    # Maximum Shifts per Nurse
    for nurse_id in nurses:
        max_shifts = max_shifts_dict[nurse_id]
        for shift_id in shift_types:
            model += (lpSum(x[(nurse_id, day_id, shift_id)] for day_id in days) <= max_shifts.get(shift_id, 0), f"max_shifts_{nurse_id}_{shift_id}")

    # Coverage constraints
    for day_id in days:
        for shift_id in shift_types:
            if (day_id, shift_id) in cover_requirements_dict:
                requirement = cover_requirements_dict[(day_id, shift_id)]
                model += (lpSum(x[(nurse_id, day_id, shift_id)] for nurse_id in nurses) >= requirement, f"min_cover_{day_id}_{shift_id}")

    # Maximum Total Working Time
    for nurse_id in nurses:
        model += (lpSum(x[(nurse_id, day_id, shift_id)] for day_id in days for shift_id in shift_types) * 480 <= max_total_minutes[nurse_id], f"max_total_minutes_{nurse_id}")

    # Consecutive Shifts
    for nurse_id in nurses:
        max_consecutive = max_consecutive_shifts[nurse_id]
        for day_index in range(len(days) - max_consecutive):
            model += (lpSum(x[(nurse_id, days[day_index + d], shift_id)] for d in range(max_consecutive) for shift_id in shift_types) <= max_consecutive, f"max_consecutive_shifts_{nurse_id}_{day_index}")

    # Hard Constraints: Days Off
    for nurse_id in nurses:
        if nurse_id in days_off_dict:
            for day_id in days_off_dict[nurse_id]:
                for shift_id in shift_types:
                    model += (x[(nurse_id, day_id, shift_id)] == 0, f"day_off_{nurse_id}_{day_id}_{shift_id}")

    # Add the constraint that each nurse can work at most one shift per day
    for nurse_id in nurses:
        for day_id in days:
            model += (lpSum(x[(nurse_id, day_id, shift_id)] for shift_id in shift_types) <= 1, f"one_shift_per_day_{nurse_id}_{day_id}")

    # Solve the model
    model.solve(PULP_CBC_CMD(msg=True, timeLimit=3600, options=["feasibilityTol", "1e-12", "maxIterations", "10000000"]))

    # Extract solution
    assigned_shifts = {(nurse_id, day_id, shift_id): var.varValue for (nurse_id, day_id, shift_id), var in x.items() if var.varValue > 0}

    # Aggregate schedule for export
    schedule_aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (nurse_id, day_id, shift_id), var in x.items():
        if var.varValue > 0:
            week_number = (day_id // 7) + 1
            day_of_week = day_id % 7
            schedule_aggregated[week_number][day_of_week][shift_id].append(nurse_id)

    return schedule_aggregated, assigned_shifts


# Feasibility checks
def check_cover_requirements_met(assigned_shifts, cover_requirements_dict, nurses, days, shift_types):
    violations = []
    cover_met = True

    for (day, shift), required in cover_requirements_dict.items():
        assigned_nurses = [nurse for (nurse, day_num, shift_type) in assigned_shifts if day_num == day and shift_type == shift]
        num_assigned = len(assigned_nurses)
        if num_assigned < required:
            violations.append(f"Cover violation on Day {day}, Shift {shift}: Assigned {num_assigned}, Required {required}")
            cover_met = False

    if violations:
        print("\nCover requirement violations:")
        for violation in violations:
            print(violation)
    else:
        print("Cover requirements: All cover requirements were met.")

    return cover_met


# Function to check if labor law constraints were respected
def check_labor_law_compliance(assigned_shifts, nurses, days):
    violations = []
    labor_law_met = True

    for nurse_id in nurses:
        for day_id in range(len(days) - 1):
            # Check Rule 1: A nurse cannot work an Early shift (E) if they worked a Day shift (D) the previous day
            if assigned_shifts.get((nurse_id, day_id, 'D'), 0) > 0 and assigned_shifts.get((nurse_id, day_id + 1, 'E'), 0) > 0:
                violations.append(f"Labor law violation: Nurse {nurse_id} worked Day shift followed by Early shift on days {day_id} and {day_id + 1}")
                labor_law_met = False

            # Check Rule 2: A nurse cannot work a Day shift (D) if they worked a Late shift (L) the previous day
            if assigned_shifts.get((nurse_id, day_id, 'L'), 0) > 0 and assigned_shifts.get((nurse_id, day_id + 1, 'D'), 0) > 0:
                violations.append(f"Labor law violation: Nurse {nurse_id} worked Late shift followed by Day shift on days {day_id} and {day_id + 1}")
                labor_law_met = False

            # Check Rule 3: A nurse cannot work an Early shift (E) if they worked a Late shift (L) the previous day
            if assigned_shifts.get((nurse_id, day_id, 'L'), 0) > 0 and assigned_shifts.get((nurse_id, day_id + 1, 'E'), 0) > 0:
                violations.append(f"Labor law violation: Nurse {nurse_id} worked Late shift followed by Early shift on days {day_id} and {day_id + 1}")
                labor_law_met = False

    if violations:
        print("\nLabor law violations:")
        for violation in violations:
            print(violation)
    else:
        print("Labor law: All labor law constraints were respected.")

    return labor_law_met


# Check if all nurses respect their days off
def check_days_off_respected(assigned_shifts, days_off_dict):
    respected_days_off = 0
    total_days_off = 0
    violations = []

    for nurse_id, days_off in days_off_dict.items():
        for day in days_off:
            total_days_off += 1
            if not any(assigned_shifts.get((nurse_id, day, shift), 0) > 0 for shift in ['E', 'D', 'L']):
                respected_days_off += 1
            else:
                violations.append(f"Nurse {nurse_id} was assigned on their requested day off (Day {day})")

    if violations:
        print("\nDays off violations:")
        for violation in violations:
            print(violation)
    else:
        print("All day-off requests were respected.")

    return respected_days_off == total_days_off


# Streamlit UI
st.title("The Girlies Nurse Scheduling System <3")

# File uploaders
staff_file = st.file_uploader("Upload the staff CSV", type="csv")
cover_requirements_file = st.file_uploader("Upload the cover requirements CSV", type="csv")
days_off_file = st.file_uploader("Upload the days off CSV", type="csv")
shift_on_requests_file = st.file_uploader("Upload the shift on requests CSV", type="csv")
shift_off_requests_file = st.file_uploader("Upload the shift off requests CSV", type="csv")

if staff_file and cover_requirements_file and days_off_file and shift_on_requests_file and shift_off_requests_file:
    staff = pd.read_csv(staff_file)
    cover_requirements = pd.read_csv(cover_requirements_file)
    days_off = pd.read_csv(days_off_file)
    shift_on_requests = pd.read_csv(shift_on_requests_file)
    shift_off_requests = pd.read_csv(shift_off_requests_file)

    with st.spinner('Creating the schedule...'):
        schedule_aggregated, assigned_shifts = create_schedule(staff, cover_requirements, days_off, shift_on_requests, shift_off_requests)
        st.success('Schedule created successfully!')

    # Get the list of days from the cover requirements for use in checks
    days = cover_requirements['# Day'].unique().tolist()
    shift_types = cover_requirements[' ShiftID'].unique().tolist()

    # Assume day 0 is today
    start_date = datetime.today()

    # Prepare events for the calendar component
    events = []
    shift_labels = {'E': 'Early shift', 'D': 'Day shift', 'L': 'Late shift'}
    
    for week_number in schedule_aggregated:
        for day_of_week in schedule_aggregated[week_number]:
            shift_day = start_date + timedelta(days=((week_number - 1) * 7 + day_of_week))
            for shift, nurses in schedule_aggregated[week_number][day_of_week].items():
                for nurse in nurses:
                    if shift == 'E':
                        start_time = "00:00:00"
                        end_time = "08:00:00"
                    elif shift == 'D':
                        start_time = "08:00:00"
                        end_time = "16:00:00"
                    elif shift == 'L':
                        start_time = "16:00:00"
                        end_time = "23:59:59"

                    event = {
                        "title": f"Nurse {nurse} - {shift_labels.get(shift, shift)}",
                        "start": f"{shift_day.strftime('%Y-%m-%d')}T{start_time}",
                        "end": f"{shift_day.strftime('%Y-%m-%d')}T{end_time}",
                    }
                    events.append(event)

    calendar_options = {
        "editable": "false",
        "selectable": "true",
        "headerToolbar": {
            "left": "today prev,next",
            "center": "title",
            "right": "dayGridMonth,timeGridWeek,timeGridDay",
        },
        "initialView": "dayGridMonth",
    }

    custom_css = """
        .fc-event-time {
            font-style: italic;
        }
        .fc-event-title {
            font-weight: bold;
        }
    """

    # Use the Calendar component to display the schedule
    calendar_component = calendar(events=events, options=calendar_options, custom_css=custom_css)
    st.write(calendar_component)

    # Option to download the schedule as CSV
    schedule_data = []
    total_days = 83  # Total number of days to be scheduled
    week_length = 7  # One week has 7 days

    for week_number in schedule_aggregated:
        for day_of_week in schedule_aggregated[week_number]:
            for shift, nurses in schedule_aggregated[week_number][day_of_week].items():
                schedule_data.append({
                    "Week": week_number,
                    "Day_of_Week": day_of_week,
                    "Shift": shift,
                    "Nurses": ', '.join(nurses)
                })

    schedule_df = pd.DataFrame(schedule_data)
    csv = schedule_df.to_csv().encode('utf-8')
    
    st.download_button(
        label="Download Schedule as CSV",
        data=csv,
        file_name='nurse_schedule.csv',
        mime='text/csv',
    )

    # Checking for feasibility of the schedule
    cover_check = check_cover_requirements_met(assigned_shifts, parse_cover_requirements(cover_requirements), staff['# ID'].tolist(), days, shift_types)
    labor_law_check = check_labor_law_compliance(assigned_shifts, staff['# ID'].tolist(), days)
    days_off_check = check_days_off_respected(assigned_shifts, parse_days_off(days_off))

    if cover_check and labor_law_check and days_off_check:
        st.success("The generated schedule is feasible and satisfies all constraints.")
    else:
        st.error("The generated schedule violates some constraints. Check the logs for details.")
else:
    st.warning("Please upload all required files to generate the schedule.")


# In[ ]:




