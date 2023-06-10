# Release Process

This process is a work-in-progress until the first major release of the project.

The process will ultimately aim to have:

- A published release date for the first caikit major release, upon the availability of the first beta release
- A well-known cycle of release dates for caikit minor and patch releases, upon the availability of the first caikit major release
- Minor releases backwards compatible with respect to the API, upon the availability of the first caikit major release

## Semantic versioning

caikit versions are expressed as x.y.z, where x is the major version, y is the minor version, and z is the patch version, following [Semantic Versioning](https://semver.org/spec/v2.0.0.html) terminology.

**Note:** One key principle of semantic versioning is that breaking API changes will only be delivered in a major release. While all care will be taken to respect this, there may be a scenario (for example, a security fix or a bug fix) where we would have to break the AI for a non major release. This would only be considered if no other alternative is available.

## Patch releases

Patch releases provide users with bug fixes and security fixes. They do not contain new features.

Patch releases should be done every second week on a Wednesday (assuming there are changes since the last release). A patch release to fix a high priority regression or security issue does not have to follow this schedule, but it is highly desirable that it is released on a Wednesday if possible.

### Cancelling a patch release

A patch release should be cancelled:

- If it falls within the week of a minor release

## Minor releases

Minor releases contain security and bug fixes as well as new features.

**Note:** They will be backwards compatible with respect to the API, upon the availability of the first caikit major release. Until then, they will break the API.

Minor releases should be done every other week in-between patch releases on a Wednesday (assuming there are changes since the last release).

Extra minor releases can be added to the schedule when needed.  However:

- Extra minor releases should only be done for important reasons (as per judgment of the maintainers) to avoid increasing the burden on organizations that choose to upgrade at every release.

**Note:** Release candidates (RC) for minor releases are not considered until the first major release of the project.

## Major releases

Major releases contain breaking changes. Such releases are rare but are sometimes necessary to allow caikit to continue to evolve in important new directions.

Major releases can be difficult to plan. With that in mind, a final release date will only be chosen and announced once the first beta version of such a release is available.

**Note:** A first caikit major release is NOT available yet.

## Security implications

Security releases do not follow any planned dates and should be done as soon as required.
