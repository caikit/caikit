# Contributing

👍🎉 First off, thank you for taking the time to contribute! 🎉👍

The following is a set of guidelines for contributing. These are just guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## What Should I Know Before I Get Started?

If you're new to GitHub and working with open source repositories, this section will be helpful. Otherwise, you can skip to learning how to [set up your dev environment](#set-up-your-dev-environment).

### Code of Conduct

This project adheres to the [Contributor Covenant](./CODE-OF-CONDUCT.md). By participating, you are expected to uphold this code.

Please report unacceptable behavior to one of the [Code Owners](./OWNERS).

### How Do I Start Contributing?

The below workflow is designed to help you begin your first contribution journey. It will guide you through creating and picking up issues, working through them, having your work reviewed, and then merging.

Help on open source projects is always welcome and there is always something that can be improved. For example, documentation (like the text you are reading now) can always use improvement, code can always be clarified, variables or functions can always be renamed or commented on, and there is always a need for more test coverage. If you see something that you think should be fixed, take ownership! Here is how you get started:

## How Can I Contribute?

When contributing, it's useful to start by looking at [issues](https://github.com/caikit/caikit/issues). After picking up an issue, writing code, or updating a document, make a pull request and your work will be reviewed and merged. If you're adding a new feature, it's best to [write an issue](https://github.com/caikit/caikit/issues/new?assignees=&labels=&template=feature_request.md&title=) first to discuss it with maintainers first.

### Reporting Bugs

This section guides you through submitting a bug report. Following these guidelines helps maintainers and the community understand your report ✏️, reproduce the behavior 💻, and find related reports 🔎.

#### How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitHub issues using the Bug Report template](https://github.com/caikit/caikit/issues/new?assignees=&labels=&template=bug_report.md&title=). Create an issue on that and provide the information suggested in the bug report issue template.

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion, including completely new features, tools, and minor improvements to existing functionality. Following these guidelines helps maintainers and the community understand your suggestion ✏️ and find related suggestions 🔎

#### How Do I Submit A (Good) Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues using the Feature Request template](https://github.com/caikit/caikit/issues/new?assignees=&labels=&template=feature_request.md&title=). Create an issue and provide the information suggested in the feature requests or user story issue template.

#### How Do I Submit A (Good) Improvement Item?

Improvements to existing functionality are tracked as [GitHub issues using the User Story template](https://github.com/caikit/caikit/issues/new?assignees=&labels=&template=user_story.md&title=). Create an issue and provide the information suggested in the feature requests or user story issue template.

## Development

### Set up your dev environment

Your dev environment is set up by [tox](https://tox.wiki/en/latest/), an environment orchestrator which allows for setting up environments for and invoking builds, unit tests, formatting, linting, etc. Install tox with:

```sh
pip install -r setup_requirements.txt
```

### Run unit tests

Running tests against all supported Python versions is as simple as:

```sh
tox
```

Running tests against a single Python version can be done with:

```sh
tox -e 3.9
```


### Code formatting

This project uses [pre-commit](https://pre-commit.com/) to enforce coding style using [black](https://github.com/psf/black).

You can invoke formatting with:
```sh
tox -e fmt
```

Coding style is enforced by the CI tests, so if not installed locally, your PR will fail until formatting has been applied.


## Your First Code Contribution

Unsure where to begin contributing? You can start by looking through these issues:

- Issues with the [`good first issue` label](https://github.com/caikit/caikit/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) - these should only require a few lines of code and are good targets if you're just starting contributing.
- Issues with the [`help wanted` label](https://github.com/caikit/caikit/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) - these range from simple to more complex, but are generally things we want but can't get to in a short time frame.

### How to contribute

To contribute to this repo, you'll use the Fork and Pull model common in many open source repositories. For details on this process, watch [how to contribute](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github).

When ready, you can create a pull request. Pull requests are often referred to as "PR". In general, we follow the standard [github pull request](https://help.github.com/en/articles/about-pull-requests) process. Follow the template to provide details about your pull request to the maintainers.

Before sending pull requests, make sure your changes pass tests.

#### Code Review

Once you've [created a pull request](#how-to-contribute), maintainers will review your code and likely make suggestions to fix before merging. It will be easier for your pull request to receive reviews if you consider the criteria the reviewers follow while working. Remember to:

- Run tests locally and ensure they pass
- Follow the project coding conventions
- Write detailed commit messages
- Break large changes into a logical series of smaller patches, which are easy to understand individually and combine to solve a broader issue

<!-- ## Releasing (Maintainers only)

The responsibility for releasing new versions of the libraries falls to the maintainers. Releases will follow standard [semantic versioning](https://semver.org/) and be hosted on [pypi](https://pypi.org/project/jtd-to-proto/). -->
