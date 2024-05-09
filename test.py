def filter_courses(current_course, courses, threshold):
    """
    Filter out courses that deviate more than the threshold from the current_course.
    """
    filtered_courses = []
    for course in courses:
        # Normalize both current_course and course to be within the range of -180 to 180
        normalized_current_course = (current_course + 180) % 360 - 180
        print("Normalized current course:", normalized_current_course)
        normalized_course = (course + 180) % 360 - 180
        print("Normalized course:", normalized_course)
        # Calculate the absolute difference between the courses
        diff = abs(normalized_course - normalized_current_course)
        # Check if the difference is within the threshold
        if diff <= threshold:
            filtered_courses.append(course)
    return filtered_courses

# Example usage:
current_course = 270
courses = [-350, -100, 0, 90, 200, 280]
threshold = 50
filtered_courses = filter_courses(current_course, courses, threshold)
print("Current course:", current_course)
print("All courses:", courses)
print("Filtered courses within the threshold of", threshold, "degrees from the current course:", filtered_courses)
